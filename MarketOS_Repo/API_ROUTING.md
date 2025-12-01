# API Routing Guide

## How Frontend Routes to Each Module

Even though all modules use the same endpoint path (`/api/process`), each module is deployed as a **separate service** with its own **unique Railway URL**.

### Service URLs

When you deploy each module to Railway, you'll get unique URLs like:

```
M2: https://studioos-m2-production.up.railway.app
M3: https://studioos-m3-production.up.railway.app
M4: https://studioos-m4-production.up.railway.app
M5: https://studioos-m5-production.up.railway.app
M6: https://studioos-m6-production.up.railway.app
```

### Frontend Configuration

In your frontend (or when integrating with Lovable), you'll need to configure the module URLs:

```typescript
// Example: src/lib/apiConfig.ts

export const MODULE_URLS = {
  M2: process.env.VITE_M2_API_URL || 'https://studioos-m2-production.up.railway.app',
  M3: process.env.VITE_M3_API_URL || 'https://studioos-m3-production.up.railway.app',
  M4: process.env.VITE_M4_API_URL || 'https://studioos-m4-production.up.railway.app',
  M5: process.env.VITE_M5_API_URL || 'https://studioos-m5-production.up.railway.app',
  M6: process.env.VITE_M6_API_URL || 'https://studioos-m6-production.up.railway.app',
};

// Usage in frontend
export async function callModule(moduleId: 'M2' | 'M3' | 'M4' | 'M5' | 'M6', prompt: string) {
  const url = MODULE_URLS[moduleId];
  const response = await fetch(`${url}/api/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  return response.json();
}
```

### Environment Variables for Frontend

Create a `.env.local` file in your frontend:

```bash
VITE_M2_API_URL=https://studioos-m2-production.up.railway.app
VITE_M3_API_URL=https://studioos-m3-production.up.railway.app
VITE_M4_API_URL=https://studioos-m4-production.up.railway.app
VITE_M5_API_URL=https://studioos-m5-production.up.railway.app
VITE_M6_API_URL=https://studioos-m6-production.up.railway.app
```

### Example API Calls

```bash
# M2 - Problem Validation
curl -X POST https://studioos-m2-production.up.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How to improve remote work productivity"}'

# M3 - Problem Understanding
curl -X POST https://studioos-m3-production.up.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Employee turnover in startups"}'

# M4 - Current Solutions
curl -X POST https://studioos-m4-production.up.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Apps for college student productivity"}'
```

### Pipeline Flow (Multiple Modules)

When running multiple modules in sequence, the frontend calls each module's URL:

```typescript
// Example: Run M2 → M3 → M4 pipeline
async function runPipeline(problemSpace: string, enabledModules: string[]) {
  const results = [];
  
  for (const moduleId of enabledModules) {
    const url = MODULE_URLS[moduleId];
    const result = await fetch(`${url}/api/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: problemSpace }),
    });
    results.push(await result.json());
  }
  
  return results;
}

// Usage
const results = await runPipeline("Employee turnover problem", ["M2", "M3", "M4"]);
```

### Health Checks

Each module also has a health endpoint:

```bash
# Check M2 health
curl https://studioos-m2-production.up.railway.app/health

# Response:
# {"status": "healthy", "module": "M2", "service": "Problem Validation"}
```

## Summary

- ✅ Each module has its own Railway URL
- ✅ All modules use the same endpoint path: `/api/process`
- ✅ Frontend routes to each module by calling its specific URL
- ✅ Configure module URLs in frontend environment variables
- ✅ When integrating with Lovable, provide the module URLs for each service

