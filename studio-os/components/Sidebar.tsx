import React, { useState } from 'react';
import { ModuleDefinition, ProjectState, ModuleStatus, GlobalState } from '../types';
import { CheckCircle, Lock, CircleDashed, AlertCircle, PlayCircle, LayoutDashboard, Plus, History, Trash2, ChevronRight, ChevronDown } from 'lucide-react';

interface SidebarProps {
  modules: ModuleDefinition[];
  globalState: GlobalState;
  onSelectModule: (id: string) => void;
  onCreateProject: () => void;
  onSwitchProject: (id: string) => void;
  onDeleteProject: (id: string, e: React.MouseEvent) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  modules,
  globalState,
  onSelectModule,
  onCreateProject,
  onSwitchProject,
  onDeleteProject
}) => {
  const [showProjects, setShowProjects] = useState(false);

  const activeProject = globalState.projects[globalState.activeProjectId];
  // Fix: Cast Object.values to ProjectState[] to explicitly tell TypeScript the type, resolving 'unknown' property access errors
  const sortedProjects = (Object.values(globalState.projects) as ProjectState[]).sort((a, b) => b.lastModified - a.lastModified);

  const getStatusIcon = (status: ModuleStatus) => {
    switch (status) {
      case ModuleStatus.COMPLETED: return <CheckCircle className="w-4 h-4 text-studio-500" />;
      case ModuleStatus.RUNNING: return <CircleDashed className="w-4 h-4 text-blue-500 animate-spin" />;
      case ModuleStatus.ERROR: return <AlertCircle className="w-4 h-4 text-red-500" />;
      case ModuleStatus.INTERRUPTED: return <AlertCircle className="w-4 h-4 text-amber-500" />;
      case ModuleStatus.READY: return <PlayCircle className="w-4 h-4 text-studio-400" />;
      case ModuleStatus.LOCKED: default: return <Lock className="w-4 h-4 text-gray-300" />;
    }
  };

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-screen sticky top-0 overflow-hidden">
      {/* Brand Header */}
      <div className="p-6 border-b border-studio-700 bg-studio-800 flex-shrink-0">
        <div className="flex items-center space-x-2 mb-2">
          <div className="w-8 h-8 bg-catalyst rounded-lg flex items-center justify-center">
            <LayoutDashboard className="text-charcoal w-5 h-5" />
          </div>
          <h1 className="text-xl font-serif font-bold text-white tracking-tight">StudioOS <span className="text-studio-200 text-sm align-top">v1</span></h1>
        </div>
        <p className="text-xs text-studio-200 font-medium uppercase tracking-wider mt-2">Venture Automation</p>
      </div>

      {/* Project Switcher / Creation */}
      <div className="p-4 border-b border-gray-100">
        <button
          onClick={() => setShowProjects(!showProjects)}
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors border border-gray-200"
        >
          <div className="flex items-center overflow-hidden">
            <History className="w-4 h-4 text-gray-500 mr-2 flex-shrink-0" />
            <div className="flex flex-col items-start overflow-hidden">
              <span className="text-xs text-gray-400 font-medium uppercase">Current Project</span>
              <span className="text-sm font-medium text-gray-800 truncate w-full text-left max-w-[160px]">
                {activeProject?.name || "Untitled"}
              </span>
            </div>
          </div>
          {showProjects ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronRight className="w-4 h-4 text-gray-400" />}
        </button>

        {/* Project List Drawer */}
        {showProjects && (
          <div className="mt-2 space-y-1 max-h-48 overflow-y-auto">
            <button
              onClick={() => { onCreateProject(); setShowProjects(false); }}
              className="w-full flex items-center px-3 py-2 text-sm font-medium text-studio-700 bg-studio-50 hover:bg-studio-100 rounded-md mb-2"
            >
              <Plus className="w-4 h-4 mr-2" /> New Venture Analysis
            </button>
            {sortedProjects.map(p => (
              <div
                key={p.id}
                onClick={() => { onSwitchProject(p.id); setShowProjects(false); }}
                className={`group w-full flex items-center justify-between px-3 py-2 text-sm rounded-md cursor-pointer ${p.id === activeProject.id ? 'bg-gray-100 text-gray-900' : 'text-gray-600 hover:bg-gray-50'
                  }`}
              >
                <span className="truncate flex-1">{p.name}</span>
                {sortedProjects.length > 1 && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onDeleteProject(p.id, e); }}
                    className="p-1 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Module Pipeline List */}
      <div className="flex-1 overflow-y-auto py-4">
        <div className="px-6 mb-2">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Research Pipeline</h2>
        </div>
        <nav className="space-y-0.5 px-3">
          {modules.map((module) => {
            const status = activeProject?.modules[module.id]?.status || ModuleStatus.LOCKED;
            const isActive = activeProject?.currentModuleId === module.id;
            const isLocked = status === ModuleStatus.LOCKED;

            return (
              <button
                key={module.id}
                onClick={() => !isLocked && onSelectModule(module.id)}
                disabled={isLocked}
                className={`w-full flex items-center px-3 py-3 text-sm font-medium rounded-md transition-colors ${isActive
                  ? 'bg-studio-100 text-studio-900'
                  : isLocked
                    ? 'text-gray-400 cursor-not-allowed'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`}
              >
                <span className="mr-3 flex-shrink-0">{getStatusIcon(status)}</span>
                <span className="truncate text-left">{module.title}</span>
              </button>
            );
          })}
        </nav>
      </div>


    </div>
  );
};