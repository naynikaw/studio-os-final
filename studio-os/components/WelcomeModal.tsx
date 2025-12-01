
import React from 'react';
import { X, Zap, Layers, MessageSquare, FileText } from 'lucide-react';

interface WelcomeModalProps {
  onClose: () => void;
}

export const WelcomeModal: React.FC<WelcomeModalProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl overflow-hidden animate-in fade-in zoom-in duration-300">
        <div className="bg-studio-50 p-8 border-b border-studio-100 text-center relative">
          <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-gray-600">
            <X className="w-6 h-6" />
          </button>
          <div className="w-16 h-16 bg-studio-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <Zap className="text-white w-8 h-8" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900">Welcome to StudioOS v1</h2>
          <p className="text-studio-700 mt-2">Your AI-Powered Venture Research Architect</p>
        </div>
        
        <div className="p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-start space-x-4">
              <div className="bg-blue-50 p-2 rounded-lg shrink-0">
                <Layers className="w-5 h-5 text-blue-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Automated Pipeline</h3>
                <p className="text-sm text-gray-500 mt-1">
                  14 interconnected modules that build upon each other, from trend detection to full pitch decks.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="bg-purple-50 p-2 rounded-lg shrink-0">
                <Zap className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Continuous Mode</h3>
                <p className="text-sm text-gray-500 mt-1">
                  Enabled by default. The system automatically triggers the next step as soon as one finishes.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="bg-amber-50 p-2 rounded-lg shrink-0">
                <MessageSquare className="w-5 h-5 text-amber-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Deep Dive Chat</h3>
                <p className="text-sm text-gray-500 mt-1">
                  Don't just read the report—chat with it. Upload files and ask follow-up questions for every module.
                </p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="bg-green-50 p-2 rounded-lg shrink-0">
                <FileText className="w-5 h-5 text-green-600" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Professional Export</h3>
                <p className="text-sm text-gray-500 mt-1">
                  Generate comprehensive, branded PDF reports including your chat history and citations.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-100 flex justify-end">
            <button
              onClick={onClose}
              className="px-6 py-2.5 bg-studio-600 text-white font-medium rounded-lg hover:bg-studio-700 transition-colors shadow-md"
            >
              Get Started
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
