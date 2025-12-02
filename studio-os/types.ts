
export enum ModuleStatus {
  LOCKED = 'LOCKED',
  READY = 'READY',
  RUNNING = 'RUNNING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
  INTERRUPTED = 'INTERRUPTED'
}

export interface ModuleDefinition {
  id: string;
  title: string;
  description: string;
  inputs: string[];
  systemPromptKey: string;
  isManualInput?: boolean;
  useThinking?: boolean;
  useGrounding?: boolean;
}

export interface GroundingSource {
  uri: string;
  title: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
  attachments?: {
    mimeType: string;
    data: string; // base64
  }[];
}

export interface ModuleVersion {
  timestamp: number;
  output: string;
  sources: GroundingSource[];
  chatHistory: ChatMessage[];
}

export interface ModuleData {
  status: ModuleStatus;
  output: string | null;
  sources: GroundingSource[];
  chatHistory: ChatMessage[]; // New: Conversational history for this module
  feedback: string | null;
  timestamp: number;
  versions: ModuleVersion[];
}

export interface ProjectState {
  id: string;
  name: string;
  theme: string;
  modules: Record<string, ModuleData>;
  currentModuleId: string;
  autoRun: boolean;
  lastModified: number;
}

export interface UserContext {
  persona: string;
  primaryGoal: string;
  secondaryGoal?: string;
  focusArea?: string;
}

export interface GlobalState {
  projects: Record<string, ProjectState>;
  activeProjectId: string;
  apiKey: string | null;
  hasSeenWelcome: boolean;
  userContext?: UserContext; // New: Onboarding context
}

export interface GeneratePayload {
  systemInstruction: string;
  prompt: string;
  history: string;
  useThinking: boolean;
  useGrounding: boolean;
}

export interface ChatPayload extends GeneratePayload {
  currentOutput: string; // The module's main output to discuss
  chatHistory: ChatMessage[];
  newAttachments?: { mimeType: string; data: string }[];
}
