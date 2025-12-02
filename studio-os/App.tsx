import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { ModuleView } from './components/ModuleView';
import { WelcomeModal } from './components/WelcomeModal';
import { MODULES, PROMPTS, CORE_SYSTEM_INSTRUCTION, APP_NAME } from './constants';
import { OnboardingFlow } from './components/OnboardingFlow';
import { ProjectState, ModuleStatus, ModuleData, GlobalState, ChatMessage, GroundingSource, UserContext } from './types';
import { generateGeminiResponse, generateChatResponse } from './services/geminiService';
import { Settings, Key, ToggleLeft, ToggleRight, Download, Info } from 'lucide-react';

// Initial factory for a fresh project
const createNewProject = (name: string = 'New Venture Concept'): ProjectState => {
  const initialModules: Record<string, ModuleData> = {};
  MODULES.forEach(m => {
    initialModules[m.id] = {
      status: m.id === 'mod-1' ? ModuleStatus.READY : ModuleStatus.LOCKED,
      output: null,
      sources: [],
      chatHistory: [], // Initialize chat history
      feedback: null,
      timestamp: 0,
      versions: []
    };
  });

  return {
    id: 'proj-' + Date.now(),
    name,
    theme: '',
    modules: initialModules,
    currentModuleId: 'mod-1',
    autoRun: true, // Default to TRUE as requested
    lastModified: Date.now()
  };
};

const sanitizeState = (state: GlobalState): GlobalState => {
  const newState = { ...state };
  Object.keys(newState.projects).forEach(pid => {
    const project = newState.projects[pid];
    let changed = false;
    Object.keys(project.modules).forEach(mid => {
      const mod = project.modules[mid];
      if (mod.status === ModuleStatus.RUNNING) {
        mod.status = ModuleStatus.INTERRUPTED;
        changed = true;
      }
      // Ensure chatHistory exists for old saves
      if (!mod.chatHistory) {
        mod.chatHistory = [];
        changed = true;
      }
    });
    if (changed) {
      project.lastModified = Date.now();
    }
  });
  return newState;
};

const App: React.FC = () => {
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [showWelcome, setShowWelcome] = useState(false);

  const loadState = (): GlobalState => {
    const saved = localStorage.getItem('studio_os_global_v1');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return sanitizeState(parsed);
      } catch (e) {
        console.error("Failed to load saved state", e);
      }
    }

    const firstProject = createNewProject();
    return {
      projects: { [firstProject.id]: firstProject },
      activeProjectId: firstProject.id,
      apiKey: null,
      hasSeenWelcome: false
    };
  };

  const [globalState, setGlobalState] = useState<GlobalState>(loadState());
  const abortControllerRef = useRef<AbortController | null>(null);

  // Persistence
  useEffect(() => {
    localStorage.setItem('studio_os_global_v1', JSON.stringify(globalState));
  }, [globalState]);

  // API Key & Welcome Modal Check
  useEffect(() => {
    const envKey = process.env.API_KEY;
    if (envKey && !globalState.apiKey) {
      setGlobalState(prev => ({ ...prev, apiKey: envKey }));
    } else {
      const storedKey = localStorage.getItem('studio_os_api_key');
      // Clear old Google keys if found
      if (storedKey && storedKey.startsWith('AIzaSy')) {
        localStorage.removeItem('studio_os_api_key');
        setShowApiKeyModal(true);
      } else if (storedKey) {
        setGlobalState(prev => ({ ...prev, apiKey: storedKey }));
      } else if (!globalState.apiKey) {
        setShowApiKeyModal(true);
      }
    }

    if (!globalState.hasSeenWelcome) {
      setShowWelcome(true);
    }
  }, []);

  // Close welcome modal
  const closeWelcome = () => {
    setShowWelcome(false);
    setGlobalState(prev => ({ ...prev, hasSeenWelcome: true }));
  };

  const handleOnboardingComplete = (context: UserContext) => {
    setGlobalState(prev => ({ ...prev, userContext: context }));
  };

  // --- Continuous Mode Waterfall Logic ---
  useEffect(() => {
    const activeProject = globalState.projects[globalState.activeProjectId];
    if (!activeProject || !activeProject.autoRun) return;

    const currentModuleDef = MODULES.find(m => m.id === activeProject.currentModuleId);
    if (!currentModuleDef) return;

    const currentModuleData = activeProject.modules[activeProject.currentModuleId];

    if (currentModuleData.status === ModuleStatus.COMPLETED) {
      const currentIndex = MODULES.findIndex(m => m.id === activeProject.currentModuleId);
      const nextModule = MODULES[currentIndex + 1];

      if (nextModule) {
        const nextModuleData = activeProject.modules[nextModule.id];
        if (nextModuleData.status === ModuleStatus.READY) {
          setTimeout(() => {
            updateProjectState(activeProject.id, (p) => ({ ...p, currentModuleId: nextModule.id }));
            runModule(activeProject.id, nextModule.id);
          }, 1500);
        }
      } else {
        updateProjectState(activeProject.id, (p) => ({ ...p, autoRun: false }));
      }
    }
  }, [globalState.projects, globalState.activeProjectId]);

  const updateProjectState = (projectId: string, updater: (p: ProjectState) => ProjectState) => {
    setGlobalState(prev => ({
      ...prev,
      projects: {
        ...prev.projects,
        [projectId]: updater(prev.projects[projectId])
      }
    }));
  };

  const handleCreateProject = () => {
    const newProj = createNewProject(`Venture Analysis ${Object.keys(globalState.projects).length + 1}`);
    setGlobalState(prev => ({
      ...prev,
      projects: { ...prev.projects, [newProj.id]: newProj },
      activeProjectId: newProj.id
    }));
  };

  const handleDeleteProject = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (Object.keys(globalState.projects).length <= 1) return;

    setGlobalState(prev => {
      const newProjects = { ...prev.projects };
      delete newProjects[id];
      const newActiveId = prev.activeProjectId === id ? Object.keys(newProjects)[0] : prev.activeProjectId;
      return {
        ...prev,
        projects: newProjects,
        activeProjectId: newActiveId
      };
    });
  };

  const handleSwitchProject = (id: string) => {
    setGlobalState(prev => ({ ...prev, activeProjectId: id }));
  };

  const handleSelectModule = (id: string) => {
    updateProjectState(globalState.activeProjectId, (p) => ({ ...p, currentModuleId: id }));
  };

  const toggleAutoRun = () => {
    updateProjectState(globalState.activeProjectId, (p) => ({ ...p, autoRun: !p.autoRun }));
  };

  const handleSaveApiKey = (key: string) => {
    localStorage.setItem('studio_os_api_key', key);
    setGlobalState(prev => ({ ...prev, apiKey: key }));
    setShowApiKeyModal(false);
  };

  const handleStopModule = (projectId: string, moduleId: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    updateProjectState(projectId, (p) => {
      const mod = p.modules[moduleId];
      return {
        ...p,
        autoRun: false,
        modules: {
          ...p.modules,
          [moduleId]: {
            ...mod,
            status: ModuleStatus.INTERRUPTED
          }
        }
      }
    });
  };

  // Helper to convert File to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          // Remove the "data:*/*;base64," part
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } else {
          reject(new Error("Failed to convert file"));
        }
      };
      reader.onerror = error => reject(error);
    });
  };

  const handleChat = async (message: string, files: File[] = []) => {
    const projectId = globalState.activeProjectId;
    const activeProject = globalState.projects[projectId];
    const moduleId = activeProject.currentModuleId;

    if (!globalState.apiKey) {
      setShowApiKeyModal(true);
      return;
    }

    // Convert files
    const attachments = await Promise.all(files.map(async f => ({
      mimeType: f.type,
      data: await fileToBase64(f)
    })));

    // Optimistic Update
    const userMsg: ChatMessage = {
      role: 'user',
      text: message,
      timestamp: Date.now(),
      attachments
    };

    updateProjectState(projectId, (p) => ({
      ...p,
      modules: {
        ...p.modules,
        [moduleId]: {
          ...p.modules[moduleId],
          chatHistory: [...(p.modules[moduleId].chatHistory || []), userMsg]
        }
      }
    }));

    try {
      // Construct context for chat
      const moduleIndex = MODULES.findIndex(m => m.id === moduleId);
      let history = `PROJECT THEME: ${activeProject.theme}\n\n`;
      for (let i = 0; i < moduleIndex; i++) {
        const modDef = MODULES[i];
        const modData = activeProject.modules[modDef.id];
        if (modData.output) {
          history += `--- START OUTPUT FROM MODULE ${modDef.id} (${modDef.title}) ---\n`;
          history += modData.output;
          history += `\n--- END OUTPUT FROM MODULE ${modDef.id} ---\n\n`;
        }
      }

      const modData = activeProject.modules[moduleId];

      const response = await generateChatResponse(globalState.apiKey, {
        systemInstruction: CORE_SYSTEM_INSTRUCTION,
        prompt: message,
        history: history,
        currentOutput: modData.output || "No output available yet.",
        chatHistory: modData.chatHistory || [],
        newAttachments: attachments,
        useThinking: false,
        useGrounding: false // Chat doesn't need to re-search usually, but could enable if needed
      });

      // Append AI Response
      const botMsg: ChatMessage = {
        role: 'model',
        text: response.text,
        timestamp: Date.now()
      };

      updateProjectState(projectId, (p) => ({
        ...p,
        modules: {
          ...p.modules,
          [moduleId]: {
            ...p.modules[moduleId],
            chatHistory: [...p.modules[moduleId].chatHistory, botMsg]
          }
        }
      }));

    } catch (error) {
      const errorMsg: ChatMessage = {
        role: 'model',
        text: `Error: ${(error as Error).message}`,
        timestamp: Date.now()
      };
      updateProjectState(projectId, (p) => ({
        ...p,
        modules: {
          ...p.modules,
          [moduleId]: {
            ...p.modules[moduleId],
            chatHistory: [...p.modules[moduleId].chatHistory, errorMsg]
          }
        }
      }));
    }
  };

  const runModule = async (projectId: string, moduleId: string, manualInput?: string) => {
    if (!globalState.apiKey) {
      setShowApiKeyModal(true);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const project = globalState.projects[projectId];
    const moduleIndex = MODULES.findIndex(m => m.id === moduleId);
    const moduleDef = MODULES[moduleIndex];

    updateProjectState(projectId, (prevProj) => {
      const currentMod = prevProj.modules[moduleId];
      const newVersions = [...currentMod.versions];

      // Archive output AND chat history
      if (currentMod.status === ModuleStatus.COMPLETED && currentMod.output) {
        newVersions.push({
          output: currentMod.output,
          sources: currentMod.sources,
          chatHistory: currentMod.chatHistory || [],
          timestamp: currentMod.timestamp
        });
      }

      const updatedProj = { ...prevProj };
      if (manualInput) updatedProj.theme = manualInput;
      if (manualInput && prevProj.name.startsWith("Venture Analysis")) updatedProj.name = manualInput.substring(0, 30) + "...";

      updatedProj.lastModified = Date.now();
      updatedProj.modules[moduleId] = {
        ...currentMod,
        status: ModuleStatus.RUNNING,
        output: null,
        chatHistory: [], // Reset chat for new run
        versions: newVersions
      };

      return updatedProj;
    });

    try {
      const effectiveTheme = manualInput || project.theme;
      let backendData = null;

      // HYBRID INTEGRATION: Call Flask Backend for specific modules
      if (['mod-2', 'mod-3', 'mod-4', 'mod-5', 'mod-6'].includes(moduleId)) {
        try {
          // Import dynamically to avoid top-level issues if file missing
          const { runBackendModule } = await import('./services/apiService');
          console.log(`Fetching external data for ${moduleId}...`);
          backendData = await runBackendModule(moduleId, effectiveTheme, globalState.apiKey, globalState.userContext);
        } catch (err) {
          console.warn("Backend integration failed, falling back to pure Gemini:", err);
          // Optional: Notify user via UI? For now, we log and proceed with pure Gemini
        }
      }

      const promptTemplate = PROMPTS[moduleDef.systemPromptKey] || `Analyze the previous context and generate the output for ${moduleDef.title}.`;
      let fullSystemInstruction = `${CORE_SYSTEM_INSTRUCTION}\n\nSPECIFIC MODULE INSTRUCTION:\n${promptTemplate}`;

      // Inject Backend Data if available
      if (backendData) {
        fullSystemInstruction += `\n\n=== VERIFIED EXTERNAL DATA ===\nThe following data was retrieved from real-world APIs (Reddit, Arxiv, etc.). USE THIS DATA AS THE PRIMARY SOURCE OF TRUTH for your analysis. Do not hallucinate if this data contradicts your assumptions.\n\n${JSON.stringify(backendData, null, 2)}\n\n=== END DATA ===\n`;
      }

      let history = `PROJECT THEME: ${effectiveTheme}\n\n`;
      for (let i = 0; i < moduleIndex; i++) {
        const modDef = MODULES[i];
        const modData = project.modules[modDef.id];
        if (modData.output) {
          history += `--- START OUTPUT FROM MODULE ${modDef.id} (${modDef.title}) ---\n`;
          history += modData.output;
          history += `\n--- END OUTPUT FROM MODULE ${modDef.id} ---\n\n`;
        }
      }

      const dynamicPrompt = manualInput
        ? `The user has provided the following theme/problem focus: "${manualInput}". Perform the analysis.`
        : `Based on the provided history of the venture research, execute the analysis for Module: ${moduleDef.title}.`;

      // Helper to extract sources from backend data
      const extractSourcesFromBackendData = (moduleId: string, data: any): GroundingSource[] => {
        if (!data) return [];
        const sources: GroundingSource[] = [];

        // 1. Standardized "citations" field (Preferred)
        if (data.citations && Array.isArray(data.citations)) {
          data.citations.forEach((cite: any) => {
            if (typeof cite === 'string') {
              // Handle string citations (e.g. from early M3/M6 versions)
              const urlMatch = cite.match(/https?:\/\/[^\s]+/);
              const uri = urlMatch ? urlMatch[0] : '#';
              const title = cite.replace(uri, '').trim() || cite;
              sources.push({ uri, title: `Source: ${title}` });
            } else if (typeof cite === 'object') {
              // Handle structured citations
              sources.push({
                uri: cite.uri || cite.url || '#',
                title: cite.title || cite.name || 'Source'
              });
            }
          });
        }

        // 2. Fallback: Module-specific legacy fields (if citations not present or empty)
        if (sources.length === 0) {
          if (moduleId === 'mod-2' && data.posts && Array.isArray(data.posts)) {
            data.posts.forEach((post: any) => {
              if (post.permalink && post.title) {
                sources.push({ uri: post.permalink, title: `Reddit: ${post.title}` });
              }
            });
          } else if (moduleId === 'mod-3') {
            if (data.tool_results && Array.isArray(data.tool_results)) {
              data.tool_results.forEach((res: any) => {
                if (res.pdf_url) sources.push({ uri: res.pdf_url, title: `Arxiv: ${res.title}` });
                if (res.url) sources.push({ uri: res.url, title: `Web: ${res.title}` });
              });
            }
          }
        } else if (moduleId === 'mod-6' && data.citations && Array.isArray(data.citations)) {
          data.citations.forEach((item: any) => {
            if (item.url && item.title) {
              sources.push({ uri: item.url, title: `News: ${item.title}` });
            }
          });
        }
        return sources;
      };

      const backendSources = extractSourcesFromBackendData(moduleId, backendData);

      const { text, sources: geminiSources } = await generateGeminiResponse(globalState.apiKey, {
        systemInstruction: fullSystemInstruction,
        prompt: dynamicPrompt,
        history: history,
        useThinking: !!moduleDef.useThinking,
        useGrounding: !!moduleDef.useGrounding && !backendData
      });

      if (abortControllerRef.current?.signal.aborted) return;

      // Merge sources: Backend sources first (verified), then Gemini sources
      const allSources = [...backendSources, ...geminiSources];

      updateProjectState(projectId, (prev) => {
        const next = { ...prev };
        next.modules[moduleId] = {
          ...next.modules[moduleId],
          status: ModuleStatus.COMPLETED,
          output: text,
          sources: allSources,
          timestamp: Date.now()
        };

        const nextModuleDef = MODULES[moduleIndex + 1];
        if (nextModuleDef) {
          next.modules[nextModuleDef.id].status = ModuleStatus.READY;
        }

        return next;
      });

    } catch (error) {
      if (abortControllerRef.current?.signal.aborted) return;

      updateProjectState(projectId, (prev) => {
        const next = { ...prev };
        next.modules[moduleId].status = ModuleStatus.ERROR;
        next.modules[moduleId].output = `Error: ${(error as Error).message}`;
        next.autoRun = false;
        return next;
      });
    } finally {
      abortControllerRef.current = null;
    }
  };

  const handlePrintReport = () => {
    const project = globalState.projects[globalState.activeProjectId];
    const printWindow = window.open('', '_blank');
    if (!printWindow) return;

    const completedModules = MODULES.filter(m => project.modules[m.id].status === ModuleStatus.COMPLETED);

    let htmlContent = `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>${project.name || APP_NAME} - Report</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
          @media print {
            @page { margin: 2cm; }
            body { -webkit-print-color-adjust: exact; }
          }
          body { 
            font-family: 'Inter', sans-serif; 
            color: #1f2937; 
            line-height: 1.7; 
            background: #fff;
            margin: 0;
            padding: 0;
          }
          .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
          }
          
          /* Cover Page */
          .cover-page {
            height: 90vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: center;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 50px;
            page-break-after: always;
          }
          .brand {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #059669; /* Studio Green */
            text-transform: uppercase;
            font-size: 14px;
            margin-bottom: 20px;
          }
          .report-title {
            font-family: 'Playfair Display', serif;
            font-size: 48px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 10px;
            line-height: 1.2;
          }
          .report-theme {
            font-size: 20px;
            color: #6b7280;
            font-weight: 300;
            margin-bottom: 40px;
          }
          .meta-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 400px;
            margin: 0 auto;
            text-align: left;
            border-top: 1px solid #e5e7eb;
            padding-top: 20px;
          }
          .meta-label { font-size: 11px; text-transform: uppercase; color: #9ca3af; letter-spacing: 0.05em; }
          .meta-value { font-size: 14px; font-weight: 500; color: #374151; }

          /* Content */
          h1 { font-size: 24px; color: #111; border-bottom: 2px solid #059669; padding-bottom: 10px; margin-top: 0; }
          h2 { font-size: 20px; color: #374151; margin-top: 30px; font-weight: 600; }
          h3 { font-size: 18px; color: #4b5563; font-weight: 600; }
          p { margin-bottom: 1.2em; }
          ul, ol { margin-bottom: 1.2em; padding-left: 1.5em; }
          li { margin-bottom: 0.5em; }
          blockquote { border-left: 4px solid #059669; padding-left: 15px; color: #4b5563; font-style: italic; margin: 20px 0; background: #f9fafb; padding: 15px; }
          
          /* Table Styles for PDF */
          table { width: 100%; border-collapse: collapse; margin-bottom: 1.5em; font-size: 0.9em; }
          th { background-color: #f3f4f6; font-weight: 600; text-align: left; padding: 8px; border: 1px solid #e5e7eb; }
          td { padding: 8px; border: 1px solid #e5e7eb; }
          tr:nth-child(even) { background-color: #f9fafb; }

          .module-section { margin-bottom: 60px; page-break-after: always; }
          .source-box { background: #f3f4f6; border-radius: 8px; padding: 15px; margin-top: 30px; font-size: 12px; }
          .source-link { color: #2563eb; text-decoration: none; display: block; margin-bottom: 4px; }
          
          /* Chat Section in PDF */
          .chat-transcript {
            margin-top: 30px;
            border-top: 1px dashed #d1d5db;
            padding-top: 20px;
          }
          .chat-header { font-size: 14px; font-weight: 700; text-transform: uppercase; color: #6b7280; margin-bottom: 15px; }
          .chat-msg { margin-bottom: 15px; font-size: 13px; }
          .chat-role { font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 2px; }
          .role-user { color: #059669; }
          .role-model { color: #4b5563; }
          .chat-text { background: #f9fafb; padding: 8px 12px; border-radius: 6px; display: inline-block; max-width: 100%; }

          /* Footer */
          .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30px;
            border-top: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            padding: 10px 40px;
            font-size: 10px;
            color: #9ca3af;
            background: white;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="cover-page">
            <div class="brand">StudioOS v1</div>
            <div class="report-title">${project.name || "Venture Research Report"}</div>
            <div class="report-theme">${project.theme}</div>
            
            <div class="meta-grid">
              <div>
                <div class="meta-label">Date Generated</div>
                <div class="meta-value">${new Date().toLocaleDateString()}</div>
              </div>
              <div>
                <div class="meta-label">Modules Completed</div>
                <div class="meta-value">${completedModules.length} / ${MODULES.length}</div>
              </div>
            </div>
          </div>

    `;

    completedModules.forEach(mod => {
      const data = project.modules[mod.id];

      // Instead of simple regex replace, we prepare containers that script will fill with marked.js output
      htmlContent += `
        <div class="module-section">
          <h1>${mod.title}</h1>
          <div id="content-${mod.id}" class="content"></div>
          <script>
            document.getElementById('content-${mod.id}').innerHTML = marked.parse(${JSON.stringify(data.output || '')});
          </script>
          
          ${data.sources && data.sources.length > 0 ? `
            <div class="source-box">
              <strong>Verified Sources:</strong>
              ${data.sources.map(s => `<a href="${s.uri}" class="source-link" target="_blank">${s.title || s.uri}</a>`).join('')}
            </div>
          ` : ''}

          ${data.chatHistory && data.chatHistory.length > 0 ? `
            <div class="chat-transcript">
              <div class="chat-header">Analyst Discussion Log</div>
              ${data.chatHistory.map(msg => `
                <div class="chat-msg">
                  <div class="chat-role ${msg.role === 'user' ? 'role-user' : 'role-model'}">${msg.role === 'user' ? 'Analyst' : 'StudioOS Agent'}</div>
                  <div class="chat-text">${msg.text}</div>
                </div>
              `).join('')}
            </div>
          ` : ''}
        </div>
      `;
    });

    htmlContent += `
        </div>
        <div class="footer">
          <span>Generated by StudioOS v1</span>
          <span>${new Date().toLocaleDateString()}</span>
        </div>
        <script>
          // Small delay to ensure marked finishes parsing before print dialog
          setTimeout(() => {
             window.print();
          }, 500);
        </script>
      </body>
      </html>
    `;

    printWindow.document.write(htmlContent);
    printWindow.document.close();
  };

  const activeProject = globalState.projects[globalState.activeProjectId];
  const currentModuleDef = MODULES.find(m => m.id === activeProject.currentModuleId)!;
  const currentModuleData = activeProject.modules[activeProject.currentModuleId];

  const canRunCurrent = currentModuleDef.inputs.every(inputId => {
    if (inputId === 'theme') return true;
    return activeProject.modules[inputId]?.status === ModuleStatus.COMPLETED;
  });

  return (
    <div className="flex h-screen bg-base-50 text-charcoal font-sans">
      {!globalState.userContext && <OnboardingFlow onComplete={handleOnboardingComplete} />}

      <Sidebar
        modules={MODULES}
        globalState={globalState}
        onSelectModule={handleSelectModule}
        onCreateProject={handleCreateProject}
        onSwitchProject={handleSwitchProject}
        onDeleteProject={handleDeleteProject}
      />

      <main className="flex-1 p-6 overflow-hidden flex flex-col">
        <header className="flex justify-between items-center mb-6 shrink-0">
          <div className="flex items-center space-x-4">
            <h3 className="text-sm font-medium text-gray-500">Workspace</h3>
            {/* Auto-Run Toggle */}
            <div className="flex items-center bg-white px-3 py-1.5 rounded-full border border-gray-200 shadow-sm">
              <span className="text-xs font-medium text-gray-600 mr-2">Continuous Mode</span>
              <button onClick={toggleAutoRun} className={`text-catalyst focus:outline-none`}>
                {activeProject.autoRun ? <ToggleRight className="w-8 h-8 fill-catalyst" /> : <ToggleLeft className="w-8 h-8 text-gray-300" />}
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowWelcome(true)}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
              title="Product Info / Welcome"
            >
              <Info className="w-5 h-5" />
            </button>
            <button
              onClick={handlePrintReport}
              className="flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              title="Print / Save as PDF"
            >
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </button>
            <button
              onClick={() => setShowApiKeyModal(true)}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
              title="API Settings"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-hidden">
          <ModuleView
            definition={currentModuleDef}
            data={currentModuleData}
            projectTheme={activeProject.theme}
            onRun={(input) => runModule(activeProject.id, currentModuleDef.id, input)}
            onStop={() => handleStopModule(activeProject.id, currentModuleDef.id)}
            onChat={handleChat}
            canRun={canRunCurrent}
          />
        </div>
      </main>

      {showWelcome && <WelcomeModal onClose={closeWelcome} />}

      {/* API Key Modal */}
      {showApiKeyModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-8 w-full max-w-md">
            <div className="flex items-center justify-center w-12 h-12 rounded-full bg-studio-100 mb-4 mx-auto">
              <Key className="w-6 h-6 text-studio-600" />
            </div>
            <h3 className="text-xl font-bold text-center text-gray-900 mb-2">Enter OpenAI API Key</h3>
            <p className="text-sm text-center text-gray-500 mb-6">
              To orchestrate the StudioOS agents, you need a valid OpenAI API key. This is stored locally in your browser.
            </p>
            <input
              type="password"
              placeholder="sk-proj-..."
              className="w-full border border-gray-300 rounded-md px-4 py-2 mb-4 focus:ring-studio-500 focus:border-studio-500"
              onChange={(e) => setGlobalState(prev => ({ ...prev, apiKey: e.target.value }))}
              value={globalState.apiKey || ''}
            />
            <div className="flex space-x-3">
              <button
                onClick={() => handleSaveApiKey(globalState.apiKey || '')}
                className="flex-1 bg-studio-600 text-white px-4 py-2 rounded-md hover:bg-studio-700 font-medium"
              >
                Save & Continue
              </button>
              {globalState.apiKey && (
                <button
                  onClick={() => setShowApiKeyModal(false)}
                  className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-200 font-medium"
                >
                  Cancel
                </button>
              )}
            </div>
            <p className="text-xs text-center text-gray-400 mt-4">
              Get a key at <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer" className="underline hover:text-studio-600">OpenAI Platform</a>
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;