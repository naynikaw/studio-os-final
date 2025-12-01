import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ModuleDefinition, ModuleData, ModuleStatus, ChatMessage } from '../types';
import { Play, RefreshCw, ChevronRight, AlertTriangle, ExternalLink, Globe, Sparkles, Square, History, Clock, AlertOctagon, Send, Paperclip, MessageSquare, User, Bot } from 'lucide-react';

interface ModuleViewProps {
  definition: ModuleDefinition;
  data: ModuleData;
  projectTheme: string;
  onRun: (manualInput?: string) => void;
  onStop: () => void;
  onChat: (message: string, files?: File[]) => Promise<void>;
  canRun: boolean;
}

export const ModuleView: React.FC<ModuleViewProps> = ({ definition, data, projectTheme, onRun, onStop, onChat, canRun }) => {
  const [manualInput, setManualInput] = React.useState(definition.isManualInput ? projectTheme : '');
  const [selectedVersionIndex, setSelectedVersionIndex] = useState<number>(-1); 
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [chatFiles, setChatFiles] = useState<File[]>([]);
  
  const bottomRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);
  const manualInputRef = useRef<HTMLTextAreaElement>(null);

  const displayOutput = selectedVersionIndex === -1 ? data.output : data.versions[selectedVersionIndex]?.output;
  const displaySources = selectedVersionIndex === -1 ? data.sources : data.versions[selectedVersionIndex]?.sources;
  const displayChat = selectedVersionIndex === -1 ? data.chatHistory : data.versions[selectedVersionIndex]?.chatHistory || [];

  useEffect(() => {
    if (definition.isManualInput) {
      setManualInput(projectTheme);
    }
  }, [definition, projectTheme]);

  useEffect(() => {
    if (data.status === ModuleStatus.RUNNING && selectedVersionIndex === -1 && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [data.output, data.status, selectedVersionIndex]);

  // Scroll to bottom of chat when new messages appear
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [displayChat, isChatting]);

  // Auto-resize textarea logic
  const adjustTextareaHeight = (ref: React.RefObject<HTMLTextAreaElement>) => {
    if (ref.current) {
      ref.current.style.height = 'auto';
      ref.current.style.height = `${ref.current.scrollHeight}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight(chatInputRef);
  }, [chatInput]);

  useEffect(() => {
    adjustTextareaHeight(manualInputRef);
  }, [manualInput]);

  const handleRun = () => {
    setSelectedVersionIndex(-1);
    onRun(definition.isManualInput ? manualInput : undefined);
  };

  const handleSendChat = async () => {
    if ((!chatInput.trim() && chatFiles.length === 0) || isChatting) return;
    
    setIsChatting(true);
    try {
      await onChat(chatInput, chatFiles);
      setChatInput('');
      setChatFiles([]);
      // Reset height
      if (chatInputRef.current) chatInputRef.current.style.height = 'auto';
    } catch (e) {
      console.error(e);
    } finally {
      setIsChatting(false);
    }
  };

  const handleChatKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendChat();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setChatFiles(prev => [...prev, ...Array.from(e.target.files || [])]);
    }
  };

  const removeFile = (index: number) => {
    setChatFiles(prev => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-8 py-6 border-b border-gray-100 bg-white shrink-0">
        <div className="flex justify-between items-start">
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-2xl font-bold text-gray-900">{definition.title}</h2>
              {definition.useThinking && (
                <span className="bg-purple-50 text-purple-700 text-xs px-2 py-1 rounded-full border border-purple-100 flex items-center">
                  <Sparkles className="w-3 h-3 mr-1" /> Thinking
                </span>
              )}
              {definition.useGrounding && (
                 <span className="bg-blue-50 text-blue-700 text-xs px-2 py-1 rounded-full border border-blue-100 flex items-center">
                 <Globe className="w-3 h-3 mr-1" /> Live Search
               </span>
              )}
            </div>
            <p className="mt-2 text-gray-500">{definition.description}</p>
          </div>
          <div className="flex items-center space-x-3">
            {/* Version Selector */}
            {data.versions.length > 0 && (
              <div className="relative group">
                <button className="inline-flex items-center px-3 py-2 border border-gray-200 text-sm font-medium rounded-md text-gray-600 bg-white hover:bg-gray-50">
                  <History className="w-4 h-4 mr-2 text-gray-400" />
                  {selectedVersionIndex === -1 ? 'Latest Version' : `Version ${data.versions.length - selectedVersionIndex}`}
                  <ChevronRight className="w-3 h-3 ml-2 transform group-hover:rotate-90 transition-transform" />
                </button>
                <div className="absolute right-0 mt-1 w-64 bg-white border border-gray-200 rounded-md shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 max-h-64 overflow-y-auto">
                  <div className="py-1">
                    <button 
                      onClick={() => setSelectedVersionIndex(-1)}
                      className={`block w-full text-left px-4 py-2 text-sm ${selectedVersionIndex === -1 ? 'bg-studio-50 text-studio-700' : 'text-gray-700 hover:bg-gray-50'}`}
                    >
                      Latest Result
                    </button>
                    {data.versions.map((v, idx) => (
                      <button 
                        key={idx}
                        onClick={() => setSelectedVersionIndex(idx)}
                        className={`block w-full text-left px-4 py-2 text-sm ${selectedVersionIndex === idx ? 'bg-studio-50 text-studio-700' : 'text-gray-700 hover:bg-gray-50'}`}
                      >
                        <div className="flex items-center">
                          <Clock className="w-3 h-3 mr-2 text-gray-400" />
                          <div className="flex flex-col">
                            <span>{new Date(v.timestamp).toLocaleTimeString()}</span>
                            <span className="text-gray-400 text-xs">{new Date(v.timestamp).toLocaleDateString()}</span>
                          </div>
                        </div>
                      </button>
                    )).reverse()}
                  </div>
                </div>
              </div>
            )}

            {/* Actions */}
            {data.status === ModuleStatus.RUNNING ? (
              <button
                onClick={onStop}
                className="inline-flex items-center px-4 py-2 border border-red-300 text-sm font-medium rounded-md text-red-700 bg-red-50 hover:bg-red-100 focus:outline-none"
              >
                <Square className="w-3 h-3 fill-current mr-2" />
                Stop
              </button>
            ) : (
              (data.status === ModuleStatus.COMPLETED || data.status === ModuleStatus.ERROR || data.status === ModuleStatus.INTERRUPTED) && (
                <button
                  onClick={handleRun}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-studio-500"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Re-run
                </button>
              )
            )}
          </div>
        </div>
      </div>

      {/* Input Area (for manual modules) */}
      {definition.isManualInput && data.status !== ModuleStatus.RUNNING && data.status !== ModuleStatus.COMPLETED && (
        <div className="px-8 py-6 bg-studio-50 border-b border-studio-100 shrink-0">
          <label className="block text-sm font-medium text-studio-900 mb-2">
            Focus Theme / Problem Area
          </label>
          <div className="flex flex-col space-y-3">
            <textarea
              ref={manualInputRef}
              value={manualInput}
              onChange={(e) => setManualInput(e.target.value)}
              placeholder="e.g. AI-driven cognitive tutors for frontline healthcare workers"
              rows={1}
              className="w-full shadow-sm focus:ring-studio-500 focus:border-studio-500 block border-gray-300 rounded-md p-3 border bg-white text-gray-900 resize-none overflow-hidden min-h-[46px]"
            />
            <div className="flex justify-end">
              <button
                onClick={handleRun}
                disabled={!manualInput.trim()}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-studio-600 hover:bg-studio-700 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Start Research <ChevronRight className="ml-2 w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Action Area (for automated modules) */}
      {!definition.isManualInput && data.status === ModuleStatus.READY && (
        <div className="px-8 py-6 bg-gray-50 border-b border-gray-200 shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center text-gray-600">
              <AlertTriangle className="w-5 h-5 text-amber-500 mr-2" />
              <span className="text-sm">This module requires input from previous steps. Ready to proceed.</span>
            </div>
            <button
              onClick={handleRun}
              disabled={!canRun}
              className="inline-flex items-center px-6 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-studio-600 hover:bg-studio-700 focus:outline-none disabled:bg-gray-300"
            >
              <Play className="w-4 h-4 mr-2" />
              Generate Analysis
            </button>
          </div>
        </div>
      )}
      
      {/* Interrupted / Error State */}
      {data.status === ModuleStatus.INTERRUPTED && (
         <div className="px-8 py-4 bg-amber-50 border-b border-amber-100 shrink-0 flex items-center text-amber-800">
            <AlertOctagon className="w-5 h-5 mr-2" />
            <span className="text-sm font-medium">Analysis was interrupted (power loss or reload). Use "Re-run" to restart.</span>
         </div>
      )}

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto bg-white relative scroll-smooth">
        <div className="p-8">
          {data.status === ModuleStatus.RUNNING && !displayOutput && (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500 animate-pulse">
              <div className="w-12 h-12 border-4 border-studio-200 border-t-studio-600 rounded-full animate-spin mb-4"></div>
              <p>Orchestrating research agents...</p>
              {definition.useGrounding && <p className="text-xs mt-2 text-blue-500">Browsing live web sources & validating...</p>}
              {definition.useThinking && <p className="text-xs mt-1 text-purple-500">Reasoning deeply about market dynamics...</p>}
            </div>
          )}

          {displayOutput ? (
            <>
              {selectedVersionIndex !== -1 && (
                  <div className="mb-6 p-3 bg-gray-100 rounded text-xs text-gray-500 text-center">
                    Viewing historical version from {new Date(data.versions[selectedVersionIndex].timestamp).toLocaleString()}
                  </div>
              )}
              <div className="prose prose-studio max-w-none mb-12">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayOutput}</ReactMarkdown>
              </div>

              {/* Sources/Citations Footer */}
              {displaySources && displaySources.length > 0 && (
                <div className="mt-8 pt-6 border-t border-gray-100">
                  <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 flex items-center">
                    <Globe className="w-3 h-3 mr-1" /> Verified Sources
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {displaySources.map((source, idx) => (
                      <a 
                        key={idx} 
                        href={source.uri} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center p-2 rounded-md hover:bg-gray-50 border border-transparent hover:border-gray-200 group transition-all"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-700 truncate group-hover:text-studio-700">
                            {source.title || new URL(source.uri).hostname}
                          </p>
                          <p className="text-xs text-gray-400 truncate">{source.uri}</p>
                        </div>
                        <ExternalLink className="w-3 h-3 text-gray-300 group-hover:text-studio-500 ml-2" />
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* CHAT INTERFACE SECTION */}
              <div className="mt-12 border-t-2 border-gray-100 pt-8 pb-4">
                <div className="flex items-center mb-6">
                  <MessageSquare className="w-5 h-5 text-studio-600 mr-2" />
                  <h3 className="text-lg font-bold text-gray-800">Deep Dive & Discussion</h3>
                </div>
                
                <div className="bg-gray-50 rounded-xl p-4 min-h-[200px] mb-4 border border-gray-200">
                  {displayChat.length === 0 ? (
                    <div className="text-center py-8 text-gray-400 text-sm">
                      <p>Have questions about this analysis?</p>
                      <p>Start a conversation to explore deeper, ask for clarifications, or refine the output.</p>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {displayChat.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                            msg.role === 'user' 
                              ? 'bg-studio-600 text-white rounded-br-none' 
                              : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none shadow-sm'
                          }`}>
                            {/* Attachments */}
                            {msg.attachments && msg.attachments.length > 0 && (
                               <div className="flex gap-2 mb-2 overflow-x-auto">
                                 {msg.attachments.map((att, i) => (
                                   <div key={i} className="text-xs bg-black/20 p-1 rounded text-white flex items-center">
                                     <Paperclip className="w-3 h-3 mr-1" /> Attachment
                                   </div>
                                 ))}
                               </div>
                            )}
                            <div className={`text-sm ${msg.role === 'user' ? 'text-white' : 'prose prose-sm max-w-none'}`}>
                              {msg.role === 'user' ? msg.text : <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>}
                            </div>
                            <div className={`text-[10px] mt-1 opacity-70 ${msg.role === 'user' ? 'text-studio-100' : 'text-gray-400'}`}>
                              {new Date(msg.timestamp).toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                      ))}
                      {isChatting && (
                         <div className="flex justify-start">
                            <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 rounded-bl-none shadow-sm">
                              <div className="flex space-x-1">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                              </div>
                            </div>
                         </div>
                      )}
                      <div ref={chatEndRef} />
                    </div>
                  )}
                </div>

                {/* Chat Input */}
                {selectedVersionIndex === -1 && (
                  <div className="relative">
                    {chatFiles.length > 0 && (
                      <div className="flex gap-2 mb-2 flex-wrap">
                        {chatFiles.map((f, i) => (
                          <div key={i} className="bg-studio-50 text-studio-700 text-xs px-2 py-1 rounded-md border border-studio-200 flex items-center">
                            <Paperclip className="w-3 h-3 mr-1" /> {f.name}
                            <button onClick={() => removeFile(i)} className="ml-2 text-studio-400 hover:text-studio-600"><div className="w-3 h-3">×</div></button>
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="flex gap-2 items-end">
                      <button 
                        onClick={() => fileInputRef.current?.click()}
                        className="p-3 text-gray-400 hover:text-gray-600 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors h-[50px]"
                        title="Upload file"
                      >
                        <Paperclip className="w-5 h-5" />
                      </button>
                      <input 
                        type="file" 
                        ref={fileInputRef} 
                        onChange={handleFileSelect} 
                        className="hidden" 
                        multiple 
                        accept="image/*,application/pdf,.txt"
                      />
                      <div className="flex-1 relative">
                        <textarea
                          ref={chatInputRef}
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyDown={handleChatKeyDown}
                          placeholder="Ask a follow-up question..."
                          rows={1}
                          className="w-full border border-gray-300 rounded-lg pl-4 pr-12 py-3 focus:ring-studio-500 focus:border-studio-500 shadow-sm bg-white text-gray-900 resize-none overflow-hidden min-h-[50px]"
                          disabled={isChatting}
                        />
                        <button
                          onClick={handleSendChat}
                          disabled={(!chatInput.trim() && chatFiles.length === 0) || isChatting}
                          className="absolute right-2 bottom-2 p-1.5 bg-studio-600 text-white rounded-md hover:bg-studio-700 disabled:opacity-50 transition-colors mb-1"
                        >
                          <Send className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>

            </>
          ) : data.status === ModuleStatus.ERROR ? (
            <div className="text-red-500 p-4 bg-red-50 rounded-lg">
              Error generating content. Please try re-running the module.
            </div>
          ) : null}
          
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
};