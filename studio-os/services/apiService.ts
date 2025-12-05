import { ModuleData, UserContext } from '../types';
import { ENABLE_USER_CONTEXT } from '../constants';

// Use environment variable for production, fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface BackendResponse {
    success: boolean;
    data: any;
    message?: string;
}

export const runBackendModule = async (moduleId: string, prompt: string, apiKey?: string, userContext?: UserContext): Promise<any> => {
    // Map module IDs to endpoints
    const endpointMap: Record<string, string> = {
        'mod-2': '/api/m2',
        'mod-3': '/api/m3',
        'mod-4': '/api/m4',
        'mod-5': '/api/m5',
        'mod-6': '/api/m6',
    };

    const endpoint = endpointMap[moduleId];
    if (!endpoint) {
        throw new Error(`No backend endpoint for module ${moduleId}`);
    }

    const payload: any = { prompt, apiKey };
    if (ENABLE_USER_CONTEXT && userContext) {
        payload.user_context = userContext;
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Backend error: ${response.statusText}`);
        }

        const result: BackendResponse = await response.json();
        if (!result.success) {
            throw new Error(result.message || 'Unknown backend error');
        }

        return result.data;
    } catch (error) {
        console.error(`Failed to run backend module ${moduleId}:`, error);
        throw error;
    }
};
