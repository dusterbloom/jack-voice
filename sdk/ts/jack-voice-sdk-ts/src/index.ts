import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { createInterface, Interface as ReadLineInterface } from "node:readline";

type JsonObject = Record<string, unknown>;

interface PendingRequest {
  method: string;
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
  timer?: NodeJS.Timeout;
}

interface BridgeResponse extends JsonObject {
  type: "response";
  id: string;
  ok: boolean;
  result?: unknown;
  error?: unknown;
}

export interface BridgeEvent extends JsonObject {
  type: "event";
  event: string;
  session_id?: string;
  ts_ms?: number;
  data?: unknown;
}

export interface ConnectOptions {
  bridgePath?: string;
  bridgeArgs?: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  autoConfigure?: boolean;
  requestTimeoutMs?: number;
  helloParams?: Record<string, unknown>;
  onEvent?: (event: BridgeEvent) => void;
}

export interface BaseRequestOptions {
  timeoutMs?: number;
  [key: string]: unknown;
}

export interface AudioRequestOptions extends BaseRequestOptions {
  format?: "pcm_s16le" | "f32le";
  sampleRateHz?: number;
  channels?: number;
}

export interface TtsOptions extends BaseRequestOptions {}

export type AudioInput = Buffer | ArrayBuffer | ArrayBufferView;
export type VadResult = Record<string, unknown>;
export type SttResult = Record<string, unknown>;
export type TtsResult = Record<string, unknown>;

export class JackVoiceError extends Error {
  readonly code?: string;
  readonly retryable?: boolean;
  readonly details?: unknown;

  constructor(
    message: string,
    options: {
      code?: string;
      retryable?: boolean;
      details?: unknown;
      cause?: unknown;
    } = {},
  ) {
    super(message);
    this.name = "JackVoiceError";
    this.code = options.code;
    this.retryable = options.retryable;
    this.details = options.details;
    if (options.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }

  static fromBridgeError(method: string, payload: unknown): JackVoiceError {
    if (!isObject(payload)) {
      return new JackVoiceError(`Bridge returned an error for ${method}`);
    }

    const code = typeof payload.code === "string" ? payload.code : undefined;
    const message =
      typeof payload.message === "string"
        ? payload.message
        : `Bridge returned an error for ${method}`;
    const retryable =
      typeof payload.retryable === "boolean" ? payload.retryable : undefined;
    const details = Object.prototype.hasOwnProperty.call(payload, "details")
      ? payload.details
      : payload;

    return new JackVoiceError(message, { code, retryable, details });
  }
}

export class JackVoice {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly stdoutReader: ReadLineInterface;
  private readonly stderrReader: ReadLineInterface;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly requestTimeoutMs: number;
  private readonly onEvent?: (event: BridgeEvent) => void;
  private readonly stderrTail: string[] = [];
  private requestSeq = 0;
  private closed = false;
  private closePromise: Promise<void> | null = null;

  private constructor(
    child: ChildProcessWithoutNullStreams,
    requestTimeoutMs: number,
    onEvent?: (event: BridgeEvent) => void,
  ) {
    this.child = child;
    this.requestTimeoutMs = requestTimeoutMs;
    this.onEvent = onEvent;

    this.stdoutReader = createInterface({
      input: this.child.stdout,
      crlfDelay: Infinity,
    });
    this.stderrReader = createInterface({
      input: this.child.stderr,
      crlfDelay: Infinity,
    });

    this.stdoutReader.on("line", (line) => this.handleStdoutLine(line));
    this.stderrReader.on("line", (line) => this.handleStderrLine(line));

    this.child.on("error", (error) => {
      this.handleFatal(
        new JackVoiceError("Failed to start jack-voice-bridge", {
          code: "SPAWN_ERROR",
          cause: error,
        }),
      );
    });

    this.child.on("exit", (code, signal) => this.handleExit(code, signal));
  }

  static async connect(options: ConnectOptions = {}): Promise<JackVoice> {
    const {
      bridgePath,
      bridgeArgs = [],
      cwd,
      env,
      autoConfigure = true,
      requestTimeoutMs = 30_000,
      helloParams = {},
      onEvent,
    } = options;

    const invocation = resolveBridgeInvocation({
      bridgePath,
      bridgeArgs,
      cwd,
      env,
    });
    const procEnv = buildProcessEnv({
      baseEnv: env,
      bridgePath: invocation.bridgePath,
      cwd,
      autoConfigure,
    });
    const commandLabel = [invocation.bridgePath, ...invocation.bridgeArgs].join(" ");

    const child = spawn(invocation.bridgePath, invocation.bridgeArgs, {
      cwd,
      env: procEnv,
      stdio: ["pipe", "pipe", "pipe"],
    }) as ChildProcessWithoutNullStreams;

    const client = new JackVoice(child, requestTimeoutMs, onEvent);

    try {
      await client.request("runtime.hello", helloParams, requestTimeoutMs);
      return client;
    } catch (error) {
      await client.close().catch(() => undefined);
      if (error instanceof JackVoiceError) {
        throw error;
      }
      throw new JackVoiceError("Failed to connect to jack-voice-bridge", {
        code: "CONNECT_FAILED",
        cause: error,
        details: { command: commandLabel },
      });
    }
  }

  async vad(
    audio: AudioInput,
    options: AudioRequestOptions = {},
  ): Promise<VadResult> {
    const { timeoutMs, ...audioOptions } = options;
    const params = this.buildAudioParams(audio, audioOptions);
    return this.request<VadResult>("vad.detect", params, timeoutMs);
  }

  async stt(
    audio: AudioInput,
    options: AudioRequestOptions = {},
  ): Promise<SttResult> {
    const { timeoutMs, ...audioOptions } = options;
    const params = this.buildAudioParams(audio, audioOptions);
    return this.request<SttResult>("stt.transcribe", params, timeoutMs);
  }

  async tts(text: string, options: TtsOptions = {}): Promise<TtsResult> {
    if (!text) {
      throw new JackVoiceError("tts text must not be empty", {
        code: "INVALID_INPUT",
      });
    }

    const { timeoutMs, ...rest } = options;
    const params: Record<string, unknown> = { text, ...rest };
    return this.request<TtsResult>("tts.synthesize", params, timeoutMs);
  }

  async close(): Promise<void> {
    if (this.closePromise) {
      return this.closePromise;
    }

    this.closePromise = (async () => {
      if (!this.closed) {
        try {
          await this.request("runtime.shutdown", {}, 1_000);
        } catch {
          // Ignore if runtime is already closing or gone.
        }
      }

      if (this.child.stdin.writable) {
        this.child.stdin.end();
      }

      if (!this.child.killed && this.child.exitCode === null) {
        this.child.kill("SIGTERM");
      }

      try {
        await this.waitForExit(1_500);
      } catch {
        if (!this.child.killed && this.child.exitCode === null) {
          this.child.kill("SIGKILL");
          await this.waitForExit(500).catch(() => undefined);
        }
      }
    })();

    return this.closePromise;
  }

  private async request<TResult>(
    method: string,
    params: Record<string, unknown>,
    timeoutMsOverride?: number,
  ): Promise<TResult> {
    this.ensureActive();

    const id = this.nextRequestId();
    const timeoutMs = timeoutMsOverride ?? this.requestTimeoutMs;
    const message: Record<string, unknown> = {
      type: "request",
      id,
      method,
      params,
    };

    if (timeoutMs > 0) {
      message.timeout_ms = timeoutMs;
    }

    const responsePromise = new Promise<TResult>((resolve, reject) => {
      const pending: PendingRequest = {
        method,
        resolve: (value) => resolve(value as TResult),
        reject,
      };

      if (timeoutMs > 0) {
        pending.timer = setTimeout(() => {
          this.pending.delete(id);
          reject(
            new JackVoiceError(
              `Request timed out after ${timeoutMs}ms: ${method}`,
              {
                code: "REQUEST_TIMEOUT",
                details: { id, method, timeoutMs },
              },
            ),
          );
        }, timeoutMs);
      }

      this.pending.set(id, pending);
    });

    try {
      await this.writeLine(message);
    } catch (error) {
      this.rejectPending(
        id,
        new JackVoiceError(`Failed to send request: ${method}`, {
          code: "WRITE_FAILED",
          cause: error,
        }),
      );
    }

    return responsePromise;
  }

  private async writeLine(message: Record<string, unknown>): Promise<void> {
    if (!this.child.stdin.writable || this.child.exitCode !== null) {
      throw new JackVoiceError("Bridge stdin is not writable", {
        code: "BRIDGE_CLOSED",
      });
    }

    const line = `${JSON.stringify(message)}\n`;
    await new Promise<void>((resolve, reject) => {
      this.child.stdin.write(line, "utf8", (error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      });
    });
  }

  private handleStdoutLine(line: string): void {
    if (!line.trim()) {
      return;
    }

    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch (error) {
      this.handleFatal(
        new JackVoiceError("Received invalid JSON from bridge stdout", {
          code: "PROTOCOL_ERROR",
          cause: error,
          details: { line },
        }),
      );
      return;
    }

    if (!isObject(message) || typeof message.type !== "string") {
      return;
    }

    if (message.type === "response") {
      this.handleResponse(message as BridgeResponse);
      return;
    }

    if (message.type === "event" && this.onEvent) {
      try {
        this.onEvent(message as BridgeEvent);
      } catch {
        // Callback errors must not break protocol processing.
      }
    }
  }

  private handleResponse(message: BridgeResponse): void {
    if (typeof message.id !== "string") {
      return;
    }

    const pending = this.pending.get(message.id);
    if (!pending) {
      return;
    }

    if (pending.timer) {
      clearTimeout(pending.timer);
    }
    this.pending.delete(message.id);

    if (message.ok) {
      pending.resolve(message.result);
      return;
    }

    pending.reject(JackVoiceError.fromBridgeError(pending.method, message.error));
  }

  private handleStderrLine(line: string): void {
    if (!line) {
      return;
    }

    this.stderrTail.push(line);
    if (this.stderrTail.length > 20) {
      this.stderrTail.shift();
    }
  }

  private handleExit(
    code: number | null,
    signal: NodeJS.Signals | null,
  ): void {
    if (this.closed) {
      return;
    }

    this.closed = true;
    this.stdoutReader.close();
    this.stderrReader.close();

    const error = new JackVoiceError(
      `jack-voice-bridge exited (code=${code ?? "null"}, signal=${
        signal ?? "null"
      })`,
      {
        code: "BRIDGE_EXIT",
        details:
          this.stderrTail.length > 0 ? { stderr: [...this.stderrTail] } : undefined,
      },
    );
    this.rejectAll(error);
  }

  private handleFatal(error: JackVoiceError): void {
    if (this.closed) {
      return;
    }

    this.closed = true;
    this.stdoutReader.close();
    this.stderrReader.close();
    this.rejectAll(error);

    if (!this.child.killed && this.child.exitCode === null) {
      this.child.kill("SIGTERM");
    }
  }

  private rejectPending(id: string, error: JackVoiceError): void {
    const pending = this.pending.get(id);
    if (!pending) {
      return;
    }

    if (pending.timer) {
      clearTimeout(pending.timer);
    }
    this.pending.delete(id);
    pending.reject(error);
  }

  private rejectAll(error: JackVoiceError): void {
    for (const [id, pending] of this.pending.entries()) {
      if (pending.timer) {
        clearTimeout(pending.timer);
      }
      this.pending.delete(id);
      pending.reject(error);
    }
  }

  private ensureActive(): void {
    if (this.closed || this.child.exitCode !== null) {
      throw new JackVoiceError("jack-voice-bridge is closed", {
        code: "BRIDGE_CLOSED",
      });
    }
  }

  private nextRequestId(): string {
    this.requestSeq += 1;
    return `req_${Date.now()}_${this.requestSeq}`;
  }

  private buildAudioParams(
    audio: AudioInput,
    options: Omit<AudioRequestOptions, "timeoutMs">,
  ): Record<string, unknown> {
    const {
      format = "pcm_s16le",
      sampleRateHz = 16_000,
      channels = 1,
      ...rest
    } = options;

    return {
      ...rest,
      audio_b64: audioToBase64(audio),
      format,
      sample_rate_hz: sampleRateHz,
      channels,
    };
  }

  private async waitForExit(timeoutMs: number): Promise<void> {
    if (this.closed || this.child.exitCode !== null) {
      return;
    }

    await Promise.race([
      once(this.child, "exit").then(() => undefined),
      delay(timeoutMs).then(() => {
        throw new JackVoiceError(
          `jack-voice-bridge did not exit within ${timeoutMs}ms`,
          {
            code: "CLOSE_TIMEOUT",
          },
        );
      }),
    ]);
  }
}

export async function connect(options: ConnectOptions = {}): Promise<JackVoice> {
  return JackVoice.connect(options);
}

function resolveBridgeInvocation(options: {
  bridgePath?: string;
  bridgeArgs: string[];
  cwd?: string;
  env?: NodeJS.ProcessEnv;
}): { bridgePath: string; bridgeArgs: string[] } {
  const mergedEnv: NodeJS.ProcessEnv = { ...process.env, ...options.env };

  if (options.bridgePath && options.bridgePath.trim()) {
    return {
      bridgePath: options.bridgePath,
      bridgeArgs: [...options.bridgeArgs],
    };
  }

  const envCommand = String(mergedEnv.JACK_VOICE_BRIDGE_CMD ?? "").trim();
  if (envCommand) {
    const parsed = splitShellWords(envCommand);
    if (parsed.length > 0) {
      return {
        bridgePath: parsed[0],
        bridgeArgs: [...parsed.slice(1), ...options.bridgeArgs],
      };
    }
  }

  return {
    bridgePath: discoverBridgeBinary(options.cwd),
    bridgeArgs: [...options.bridgeArgs],
  };
}

function buildProcessEnv(options: {
  baseEnv?: NodeJS.ProcessEnv;
  bridgePath: string;
  cwd?: string;
  autoConfigure: boolean;
}): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, ...options.baseEnv };
  if (!options.autoConfigure) {
    return env;
  }

  const runtimeDirs = discoverRuntimeDirs({
    bridgePath: options.bridgePath,
    cwd: options.cwd,
    env,
  });
  const loaderVar = loaderSearchVar();
  env[loaderVar] = prependPaths(env[loaderVar], runtimeDirs);

  if (isWsl() && !env.PULSE_SERVER && existsSync("/mnt/wslg/PulseServer")) {
    env.PULSE_SERVER = "unix:/mnt/wslg/PulseServer";
  }

  return env;
}

function discoverBridgeBinary(cwd?: string): string {
  for (const candidate of candidateBridgeBinaries(cwd)) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  const pathCandidate = whichInPath(bridgeBinaryName(), process.env.PATH);
  if (pathCandidate) {
    return pathCandidate;
  }

  // Keep spawn PATH lookup behavior as fallback.
  return bridgeBinaryName();
}

function discoverRuntimeDirs(options: {
  bridgePath: string;
  cwd?: string;
  env: NodeJS.ProcessEnv;
}): string[] {
  const dirs: string[] = [];
  const resolvedExecutable = resolveExecutablePath(
    options.bridgePath,
    options.cwd,
    options.env,
  );
  if (resolvedExecutable) {
    dirs.push(path.dirname(resolvedExecutable));
  }

  for (const candidate of candidateRuntimeDirs(options.cwd)) {
    dirs.push(candidate);
  }

  return dedupe(dirs);
}

function resolveExecutablePath(
  executable: string,
  cwd: string | undefined,
  env: NodeJS.ProcessEnv,
): string | null {
  if (hasPathSeparators(executable) || path.isAbsolute(executable)) {
    const absolute = path.isAbsolute(executable)
      ? executable
      : path.resolve(cwd ?? process.cwd(), executable);
    return existsSync(absolute) ? absolute : null;
  }

  return whichInPath(executable, env.PATH);
}

function candidateBridgeBinaries(cwd?: string): string[] {
  const binaryName = bridgeBinaryName();
  const candidates: string[] = [];
  for (const root of candidateRoots(cwd)) {
    candidates.push(path.join(root, "target", "debug", binaryName));
    candidates.push(path.join(root, "target", "release", binaryName));
    candidates.push(
      path.join(root, "jack-voice-bridge", "target", "debug", binaryName),
    );
    candidates.push(
      path.join(root, "jack-voice-bridge", "target", "release", binaryName),
    );
  }
  return dedupe(candidates);
}

function candidateRuntimeDirs(cwd?: string): string[] {
  const candidates: string[] = [];
  for (const root of candidateRoots(cwd)) {
    for (const dir of [
      path.join(root, "target", "debug"),
      path.join(root, "target", "release"),
      path.join(root, "jack-voice-bridge", "target", "debug"),
      path.join(root, "jack-voice-bridge", "target", "release"),
    ]) {
      if (existsSync(dir)) {
        candidates.push(dir);
      }
    }
  }
  return dedupe(candidates);
}

function candidateRoots(cwd?: string): string[] {
  const roots: string[] = [];
  if (cwd) {
    roots.push(...ancestors(path.resolve(cwd), 6));
  }
  roots.push(...ancestors(process.cwd(), 6));
  roots.push(...ancestors(__dirname, 8));
  return dedupe(roots);
}

function ancestors(start: string, maxDepth: number): string[] {
  const values = [path.resolve(start)];
  let cursor = values[0];
  for (let i = 0; i < maxDepth; i += 1) {
    const parent = path.dirname(cursor);
    if (parent === cursor) {
      break;
    }
    values.push(parent);
    cursor = parent;
  }
  return values;
}

function bridgeBinaryName(): string {
  return process.platform === "win32"
    ? "jack-voice-bridge.exe"
    : "jack-voice-bridge";
}

function loaderSearchVar(): string {
  if (process.platform === "win32") {
    return "PATH";
  }
  if (process.platform === "darwin") {
    return "DYLD_LIBRARY_PATH";
  }
  return "LD_LIBRARY_PATH";
}

function prependPaths(existing: string | undefined, add: string[]): string {
  const current = (existing ?? "").split(path.delimiter).filter(Boolean);
  return dedupe([...add, ...current]).join(path.delimiter);
}

function whichInPath(executable: string, pathValue: string | undefined): string | null {
  if (!pathValue) {
    return null;
  }
  const dirs = pathValue.split(path.delimiter).filter(Boolean);
  const isWindows = process.platform === "win32";
  const pathext = (process.env.PATHEXT ?? ".EXE;.CMD;.BAT;.COM")
    .split(";")
    .filter(Boolean);

  const names = isWindows && path.extname(executable) === ""
    ? dedupe([
        executable,
        ...pathext.map((ext) => `${executable}${ext.toLowerCase()}`),
        ...pathext.map((ext) => `${executable}${ext}`),
      ])
    : [executable];

  for (const dir of dirs) {
    for (const name of names) {
      const candidate = path.join(dir, name);
      if (existsSync(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}

function hasPathSeparators(value: string): boolean {
  return value.includes("/") || value.includes("\\");
}

function splitShellWords(input: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  let escaped = false;

  for (const ch of input) {
    if (escaped) {
      current += ch;
      escaped = false;
      continue;
    }

    if (ch === "\\" && quote !== "'") {
      escaped = true;
      continue;
    }

    if (quote) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }

    if (/\s/.test(ch)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }

    current += ch;
  }

  if (escaped) {
    current += "\\";
  }
  if (current) {
    tokens.push(current);
  }

  return tokens;
}

function isWsl(): boolean {
  if (process.platform !== "linux") {
    return false;
  }
  const release = os.release().toLowerCase();
  return release.includes("microsoft");
}

function dedupe(values: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    out.push(value);
  }
  return out;
}

function audioToBase64(input: AudioInput): string {
  if (Buffer.isBuffer(input)) {
    return input.toString("base64");
  }

  if (input instanceof ArrayBuffer) {
    return Buffer.from(input).toString("base64");
  }

  if (ArrayBuffer.isView(input)) {
    return Buffer.from(input.buffer, input.byteOffset, input.byteLength).toString(
      "base64",
    );
  }

  throw new JackVoiceError("Unsupported audio input type", {
    code: "INVALID_INPUT",
  });
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
