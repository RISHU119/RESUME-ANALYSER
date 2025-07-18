
import { GoogleGenAI } from "@google/genai";
import type { JobSuggestion } from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable not set.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const PROMPT = `
You are an expert career advisor and resume analyst.
Analyze the following resume text and suggest 5 suitable job titles or specific industries.
For each suggestion, provide a brief (1-2 sentences) explanation of why it's a good fit based on the extracted skills and experiences.
The resume text is as follows:
---
{RESUME_TEXT}
---
Return your response as a valid JSON array of objects. Each object in the array should have exactly two keys: "title" (a string for the job title/industry) and "reason" (a string for the explanation).
Do not include any other text, introductory phrases, or explanations outside of the JSON array itself.
Example format:
[
  {
    "title": "Software Engineer",
    "reason": "The candidate has strong experience in Python, JavaScript, and building web applications, which are core skills for this role."
  },
  {
    "title": "Data Analyst",
    "reason": "Experience with SQL and data visualization tools makes the candidate a good fit for analyzing and interpreting data."
  }
]
`;

export const analyzeResume = async (resumeText: string): Promise<JobSuggestion[]> => {
    try {
        const fullPrompt = PROMPT.replace('{RESUME_TEXT}', resumeText);
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-preview-04-17',
            contents: fullPrompt,
            config: {
                responseMimeType: "application/json",
                temperature: 0.3,
            },
        });

        let jsonStr = response.text.trim();
        const fenceRegex = /^```(\w*)?\s*\n?(.*?)\n?\s*```$/s;
        const match = jsonStr.match(fenceRegex);
        if (match && match[2]) {
            jsonStr = match[2].trim();
        }

        const parsedData = JSON.parse(jsonStr);

        if (!Array.isArray(parsedData) || !parsedData.every(item => 'title' in item && 'reason' in item)) {
            throw new Error("AI response was not in the expected format.");
        }

        return parsedData as JobSuggestion[];

    } catch (error: any) {
        console.error("Error calling Gemini API:", error);
        if (error.message.includes("json")) {
             throw new Error("The AI returned an invalid response. Please try again.");
        }
        throw new Error("Failed to get suggestions from the AI. Please check your connection or API key and try again.");
    }
};
