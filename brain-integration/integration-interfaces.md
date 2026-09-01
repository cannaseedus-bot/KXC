# Integration Interfaces

Formal contracts between Brain System and Runtime Systems

---

## 1. MoS Orchestrator ↔ Brain Interface

### Request: Specialist Selection

**When**: MoS Orchestrator needs to route a query to a specialist

**Request**:
```typescript
interface SpecialistSelectionRequest {
  query: string;              // User query/prompt
  availableSpecialists: {
    id: string;
    name: string;
    expertise: string[];      // Known expertise domains
    successRate: number;      // Historical success (0-1)
  }[];
  context: {
    depth?: number;           // π depth (0-1)
    complexity?: number;      // Query complexity estimate (0-100)
    userHistory?: object;     // User interaction history
    previousResults?: object[];  // Recent specialist results
  };
  options?: {
    topK?: number;            // Return top K specialists (default: 3)
    minConfidence?: number;   // Minimum confidence threshold (default: 0.5)
    allowFallback?: boolean;  // Allow fallback if no match (default: true)
  };
}
```

**Response**:
```typescript
interface SpecialistSelectionResponse {
  ranking: {
    specialistId: string;
    confidence: number;       // 0-1, higher = better match
    reasoning: string;        // Why this specialist
    expectedLatency: number;  // ms
    expectedQuality: number;  // 0-1
  }[];
  
  primarySpecialist: {
    id: string;
    confidence: number;
  };
  
  fallbackSpecialist?: {
    id: string;
    confidence: number;
  };
  
  metadata: {
    executionTime: number;    // ms
    cacheHit: boolean;
    depth: number;            // π depth used for analysis
    entropyReduced: boolean;  // Did analysis reduce entropy?
  };
}
```

### Feedback: Specialist Performance

**After Execution**:
```typescript
interface SpecialistFeedback {
  query: string;
  specialistId: string;
  success: boolean;
  executionTime: number;      // ms
  confidence: number;         // 0-1, actual execution confidence
  userSatisfaction?: number;  // 0-1 if available
  errorMessage?: string;
  resultQuality?: number;     // 0-1
  
  // For brain learning
  learnUpdate: {
    specialistProfile: object;  // Updated specialist profile
    newConfidence: number;      // Refined confidence for future
  };
}
```

---

## 2. XJSON Runtime ↔ Brain Interface

### Request: Program Intent Analysis

**When**: XJSON program is submitted or user provides natural language intent

**Request**:
```typescript
interface ProgramIntentRequest {
  input: string | object;    // Natural language OR XJSON program object
  inputType: 'natural-language' | 'xjson-program';
  
  context: {
    userBackground?: string;  // User's domain expertise
    constraints?: {
      maxExecutionTime?: number;  // ms
      maxMemory?: number;         // bytes
      allowedSpecialists?: string[]; // Constrained specialists
    };
    previousPrograms?: object[];  // User's recent programs
  };
  
  options?: {
    suggestOptimizations?: boolean; // (default: true)
    showTemplates?: boolean;       // (default: true)
    topK?: number;                 // Top K suggestions (default: 3)
  };
}
```

**Response**:
```typescript
interface ProgramIntentResponse {
  intent: {
    goal: string;               // What the user wants to achieve
    dataFlow: {
      inputs: string[];
      outputs: string[];
      transforms: string[];
    };
    complexity: number;         // 0-100
    estimatedDepth: number;     // π depth needed (0-1)
  };
  
  suggestedPrograms: {
    program: object;            // XJSON program structure
    similarity: number;         // 0-1, how similar to intent
    confidence: number;         // 0-1, confidence this works
    rationale: string;
    estimatedLatency: number;   // ms
    fieldRecommendations?: object[];
  }[];
  
  templates: {
    name: string;
    description: string;
    isRecommended: boolean;
  }[];
  
  optimizations: {
    suggestion: string;
    impact: 'latency' | 'memory' | 'clarity';
    estimatedImprovement: number; // % improvement
  }[];
  
  metadata: {
    analysisDepth: number;      // π depth used
    executionTime: number;      // ms
  };
}
```

### Feedback: Program Execution Results

**After Execution**:
```typescript
interface ProgramExecutionFeedback {
  programId: string;
  userIntent: string;
  success: boolean;
  executionTime: number;      // Actual vs estimated
  estimatedTime: number;
  
  performanceMetrics: {
    memoryUsed: number;
    dataTransformed: number;  // bytes
    fieldEffectiveness: number; // 0-1
  };
  
  userFeedback?: {
    satisfied: boolean;
    suggestions: string[];
  };
  
  // For brain learning
  learnUpdate: {
    templateProfile: object;   // Updated template performance
    fieldPerformance: object;  // Which fields worked best
  };
}
```

---

## 3. Micronaut Runtime ↔ Brain Interface

### Request: Field Recommendation

**When**: User interaction occurs or visual state needs update

**Request**:
```typescript
interface FieldRecommendationRequest {
  interaction: {
    type: 'scroll' | 'drag' | 'click' | 'hover' | 'custom';
    target: string;             // Element ID or selector
    velocity?: number;          // For scroll/drag
    position?: { x: number, y: number };
  };
  
  worldState: {
    currentFields: object[];    // Currently active fields
    bodies: object[];           // Physics bodies
    constraints: object[];      // Physics constraints
  };
  
  userProfile?: {
    interactionStyle: string;   // 'fast', 'slow', 'precise', etc
    accessibilityNeeds?: string[]; // 'reduced-motion', etc
    preferences: object;        // User's field preferences
  };
  
  options?: {
    topK?: number;              // Top K recommendations
    targetConvergenceTime?: number; // ms
    smoothness?: 'high' | 'medium' | 'low';
  };
}
```

**Response**:
```typescript
interface FieldRecommendationResponse {
  primaryConfiguration: {
    fields: {
      type: 'wind' | 'attraction' | 'friction' | 'magnetic' | 'custom';
      parameters: object;       // Field-specific params
      strength: number;         // 0-1
      duration?: number;        // ms
    }[];
    expectedConvergence: number; // ms
    expectedSmoothnessScore: number; // 0-1
  };
  
  alternatives: {
    fields: object[];
    reason: string;
    expectedConvergence: number;
  }[];
  
  adaptations: {
    realtime: boolean;          // Adapt fields in real-time?
    learningUpdateRate: number; // How often to update preference model
  };
  
  metadata: {
    confidence: number;         // 0-1
    baselineComparison: {
      expectedLatency: number;  // vs current fields
      expectedSmoothnessGain: number; // %
    };
  };
}
```

### Feedback: Interaction Outcome

**After Interaction**:
```typescript
interface InteractionFeedback {
  interactionType: string;
  recommendedFields: object[];
  actualFields: object[];
  
  metrics: {
    convergenceTime: number;    // Actual vs predicted
    smoothnessScore: number;    // User-perceived smoothness (0-1)
    errorDistance: number;      // How far from target
    userSatisfaction?: number;  // 0-1 if available
  };
  
  learnUpdate: {
    userPreferenceBrain: object;  // Updated user profile
    fieldEffectiveness: object;   // Which fields worked
  };
}
```

---

## 4. Scheduler ↔ Brain Interface

### Request: Task Priority & Ordering

**When**: Tasks are queued and need scheduling

**Request**:
```typescript
interface TaskSchedulingRequest {
  tasks: {
    id: string;
    type: 'xjson' | 'specialist' | 'compute' | 'io';
    estimatedDuration: number;  // ms
    dependencies: string[];     // Task IDs it depends on
    priority: number;           // User-specified priority
    resources: {
      cpuNeeded: number;        // 0-1
      memoryNeeded: number;     // bytes
      gpuNeeded?: number;       // 0-1 if GPU available
    };
  }[];
  
  resources: {
    cpuAvailable: number;
    memoryAvailable: number;
    gpuAvailable?: number;
  };
  
  constraints: {
    maxQueueTime?: number;      // ms
    maxWallTime?: number;       // ms total
    energyBudget?: number;      // Joules
  };
  
  context: {
    userDeadline?: number;      // Unix timestamp
    systemLoad: number;         // 0-1
    priorityLevel: 'interactive' | 'batch' | 'background';
  };
}
```

**Response**:
```typescript
interface TaskSchedulingResponse {
  taskOrder: {
    taskId: string;
    scheduledTime: number;      // Unix timestamp
    estimatedDuration: number;  // ms
    parallelizable: boolean;    // Can run in parallel?
  }[];
  
  resourceAllocation: {
    taskId: string;
    cpuAllocated: number;
    memoryAllocated: number;
    gpuAllocated?: number;
  }[];
  
  predictions: {
    totalDuration: number;      // ms for all tasks
    expectedCompletion: number; // Unix timestamp
    bottleneckTask: string;     // Which task is critical path
    parallelGain: number;       // % speedup from parallelization
  };
  
  adaptations: {
    recommendGPUAcceleration: boolean;
    recommendBatching: boolean;
    recommendRescheduling: boolean;
  };
  
  metadata: {
    confidence: number;         // 0-1 in schedule accuracy
    analysisDepth: number;      // π depth used
  };
}
```

### Feedback: Task Execution Results

**After Task Completion**:
```typescript
interface TaskExecutionFeedback {
  taskId: string;
  success: boolean;
  actualDuration: number;     // vs estimated
  estimatedDuration: number;
  
  resourceUsage: {
    cpuUsed: number;
    memoryPeakUsed: number;
    gpuUsed?: number;
  };
  
  errors?: {
    type: string;
    message: string;
  };
  
  learnUpdate: {
    taskProfileBrain: object;  // Updated task characteristics
    scheduleOptimization: object; // How to improve future scheduling
  };
}
```

---

## 5. Cross-Layer Coordination

### Event: Integration Loop Update

**Emitted after all layers process**:
```typescript
interface IntegrationLoopUpdate {
  timestamp: number;
  cycleId: string;
  
  input: {
    userQuery: string;
    complexity: number;
    selectedDepth: number;
  };
  
  execution: {
    specialistSelected: string;
    programGenerated: boolean;
    fieldsApplied: object[];
    taskScheduled: boolean;
  };
  
  outcome: {
    success: boolean;
    confidence: number;
    latency: number;
    userSatisfaction?: number;
  };
  
  learning: {
    newSpecialistProfile: object;
    newProgramTemplate: object;
    newUserPreferences: object;
  };
}
```

---

## 6. Error Handling Contracts

### On Error: All Layers

```typescript
interface ErrorResponse {
  error: {
    code: string;              // e.g., 'SPECIALIST_UNAVAILABLE'
    message: string;
    recoverable: boolean;      // Can we continue?
  };
  
  fallback?: {
    recommended: string;       // Fallback strategy
    alternative: object;       // Alternative response
  };
  
  metadata: {
    layerFailed: string;       // Which layer failed
    timestamp: number;
  };
}
```

---

## Implementation Status

| Interface | Status | Phase |
|-----------|--------|-------|
| Specialist Selection | Defined | P2 |
| Program Intent | Defined | P3 |
| Field Recommendation | Defined | P4 |
| Task Scheduling | Defined | P5 |
| Cross-Layer Coordination | Defined | P5 |

---

## Version History

- **v1.0** (2026-03-15): Initial interface definitions
