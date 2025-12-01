
import { GeneratePayload, GroundingSource, ChatPayload, ChatMessage } from '../types';

const MODEL_ID = "gpt-5.1";

export const generateGeminiResponse = async (
  apiKey: string,
  payload: GeneratePayload
): Promise<{ text: string; sources: GroundingSource[] }> => {
  if (!apiKey) throw new Error("API Key is missing");

  const messages = [
    { role: "system", content: payload.systemInstruction || "You are a helpful assistant." },
    { role: "user", content: `CONTEXT AND HISTORY: \n${payload.history} \n\nYOUR CURRENT TASK: \n${payload.prompt} ` }
  ];

  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: MODEL_ID,
        messages: messages,
        temperature: 0.7
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`OpenAI API Error: ${response.status} ${response.statusText} - ${errText}`);
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || "No response generated.";

    // OpenAI doesn't return grounding sources in the same way, so we return empty or parse from text if needed.
    // For now, we assume citations are handled by the backend modules which return structured data.
    const sources: GroundingSource[] = [];

    return { text, sources };

  } catch (error) {
    console.error("Error generating content:", error);
    throw error;
  }
};

export const generateChatResponse = async (
  apiKey: string,
  payload: ChatPayload
): Promise<{ text: string; sources: GroundingSource[] }> => {
  if (!apiKey) throw new Error("API Key is missing");

  const messages = payload.history.map(msg => ({
    role: msg.role === 'user' ? 'user' : 'assistant',
    content: msg.message
  }));

  // Add current message
  messages.push({ role: 'user', content: payload.message });

  // Add system instruction if present (prepend)
  if (payload.systemInstruction) {
    messages.unshift({ role: 'system', content: payload.systemInstruction });
  }

  try {
    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: MODEL_ID,
        messages: messages,
        temperature: 0.7
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`OpenAI API Error: ${response.status} ${response.statusText} - ${errText}`);
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || "No response generated.";

    return { text, sources: [] };

  } catch (error) {
    console.error("Error generating chat response:", error);
    throw error;
  }
};
