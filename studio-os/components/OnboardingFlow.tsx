import React, { useState } from 'react';
import { UserContext } from '../types';
import { ArrowRight, Check, Sparkles, Target, Briefcase, GraduationCap, Building2, User } from 'lucide-react';

interface OnboardingFlowProps {
    onComplete: (context: UserContext) => void;
}

const PERSONAS = [
    { id: 'founder', label: 'Founder / Operator', icon: <Briefcase className="w-6 h-6" /> },
    { id: 'investor', label: 'Investor / Analyst', icon: <Target className="w-6 h-6" /> },
    { id: 'researcher', label: 'Student / Researcher', icon: <GraduationCap className="w-6 h-6" /> },
    { id: 'partner', label: 'Studio Partner', icon: <Building2 className="w-6 h-6" /> },
];

const GOALS = [
    'Validate a problem',
    'Explore ideas',
    'Understand a market',
    'Compare existing solutions',
    'Run a modular workflow'
];

const FOCUS_AREAS = [
    'SaaS', 'Fintech', 'Healthcare', 'Consumer', 'Deep Tech', 'Climate'
];

export const OnboardingFlow: React.FC<OnboardingFlowProps> = ({ onComplete }) => {
    const [step, setStep] = useState(1);
    const [context, setContext] = useState<Partial<UserContext>>({});
    const [customFocus, setCustomFocus] = useState('');

    const handleNext = () => {
        if (step === 4) {
            onComplete({
                persona: context.persona!,
                primaryGoal: context.primaryGoal!,
                secondaryGoal: context.secondaryGoal,
                focusArea: customFocus || context.focusArea,
            });
        } else {
            setStep(prev => prev + 1);
        }
    };

    const isStepValid = () => {
        if (step === 1) return true;
        if (step === 2) return !!context.persona;
        if (step === 3) return !!context.primaryGoal;
        return true; // Step 4 is optional
    };

    return (
        <div className="fixed inset-0 bg-base-50 z-50 flex flex-col items-center justify-center p-6 animate-in fade-in duration-300">
            <div className="w-full max-w-2xl">
                {/* Progress */}
                <div className="flex justify-between mb-12 px-2">
                    {[1, 2, 3, 4].map((s) => (
                        <div key={s} className={`h-1 flex-1 mx-1 rounded-full transition-all duration-500 ${s <= step ? 'bg-studio-600' : 'bg-gray-200'}`} />
                    ))}
                </div>

                {/* Content */}
                <div className="min-h-[400px] flex flex-col justify-center">
                    {step === 1 && (
                        <div className="text-center space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                            <div className="w-16 h-16 bg-studio-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                                <Sparkles className="w-8 h-8 text-studio-600" />
                            </div>
                            <h1 className="text-4xl font-serif font-bold text-charcoal">Welcome to StudioOS</h1>
                            <p className="text-xl text-gray-600 max-w-lg mx-auto">
                                Your AI-powered venture architect. Let's set up your workspace to tailor the analysis to your needs.
                            </p>
                        </div>
                    )}

                    {step === 2 && (
                        <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-500">
                            <div className="text-center">
                                <h2 className="text-3xl font-serif font-bold text-charcoal mb-3">Who are you?</h2>
                                <p className="text-gray-500">Select the persona that best describes your role.</p>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                {PERSONAS.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => setContext(prev => ({ ...prev, persona: p.id }))}
                                        className={`p-6 rounded-xl border-2 text-left transition-all ${context.persona === p.id
                                                ? 'border-studio-600 bg-studio-50 ring-1 ring-studio-600'
                                                : 'border-gray-200 bg-white hover:border-studio-300 hover:bg-gray-50'
                                            }`}
                                    >
                                        <div className={`mb-4 ${context.persona === p.id ? 'text-studio-600' : 'text-gray-400'}`}>
                                            {p.icon}
                                        </div>
                                        <div className="font-semibold text-charcoal">{p.label}</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {step === 3 && (
                        <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-500">
                            <div className="text-center">
                                <h2 className="text-3xl font-serif font-bold text-charcoal mb-3">What is your primary goal?</h2>
                                <p className="text-gray-500">We'll prioritize modules that help you achieve this.</p>
                            </div>
                            <div className="space-y-3">
                                {GOALS.map((goal) => (
                                    <button
                                        key={goal}
                                        onClick={() => setContext(prev => ({ ...prev, primaryGoal: goal }))}
                                        className={`w-full p-4 rounded-lg border-2 text-left transition-all flex items-center justify-between ${context.primaryGoal === goal
                                                ? 'border-studio-600 bg-studio-50'
                                                : 'border-gray-200 bg-white hover:border-studio-300'
                                            }`}
                                    >
                                        <span className="font-medium text-charcoal">{goal}</span>
                                        {context.primaryGoal === goal && <Check className="w-5 h-5 text-studio-600" />}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {step === 4 && (
                        <div className="space-y-8 animate-in slide-in-from-bottom-4 duration-500">
                            <div className="text-center">
                                <h2 className="text-3xl font-serif font-bold text-charcoal mb-3">Any specific focus area?</h2>
                                <p className="text-gray-500">Optional. Helps us ground the research.</p>
                            </div>

                            <div className="flex flex-wrap gap-3 justify-center mb-6">
                                {FOCUS_AREAS.map((area) => (
                                    <button
                                        key={area}
                                        onClick={() => setContext(prev => ({ ...prev, focusArea: area }))}
                                        className={`px-4 py-2 rounded-full border transition-all ${context.focusArea === area
                                                ? 'bg-studio-600 text-white border-studio-600'
                                                : 'bg-white text-gray-600 border-gray-200 hover:border-studio-300'
                                            }`}
                                    >
                                        {area}
                                    </button>
                                ))}
                            </div>

                            <div className="max-w-md mx-auto">
                                <label className="block text-sm font-medium text-gray-700 mb-2">Or type a custom theme</label>
                                <input
                                    type="text"
                                    value={customFocus}
                                    onChange={(e) => {
                                        setCustomFocus(e.target.value);
                                        setContext(prev => ({ ...prev, focusArea: undefined })); // Clear selection if typing
                                    }}
                                    placeholder="e.g. Generative Biology"
                                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-studio-500 focus:border-studio-500 outline-none"
                                />
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer Actions */}
                <div className="mt-12 flex justify-center">
                    <button
                        onClick={handleNext}
                        disabled={!isStepValid()}
                        className="group flex items-center px-8 py-4 bg-catalyst text-charcoal font-bold rounded-full shadow-lg hover:bg-catalyst-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105"
                    >
                        {step === 1 ? "Let's set up your workspace" : step === 4 ? "Enter StudioOS" : "Continue"}
                        <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </button>
                </div>

                {step > 1 && (
                    <button
                        onClick={() => setStep(prev => prev - 1)}
                        className="mt-6 mx-auto block text-sm text-gray-400 hover:text-gray-600"
                    >
                        Back
                    </button>
                )}
            </div>
        </div>
    );
};
