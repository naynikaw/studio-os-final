import { ModuleData } from '../types';

const API_BASE_URL = 'http://localhost:8000'; // Default local Flask port

export interface BackendResponse {
    success: boolean;
    data: any;
    message?: string;
}

export const runBackendModule = async (moduleId: string, prompt: string, apiKey?: string): Promise<any> => {
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

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, apiKey }),
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
