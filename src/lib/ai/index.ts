/**
 * Enhanced AI Configuration with 1M Context and 128K Input Tokens
 */

export interface AIModelConfig {
  id: string;
  name: string;
  maxContextTokens: number;
  maxInputTokens: number;
  maxOutputTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  presencePenalty: number;
  frequencyPenalty: number;
  responseOptimization: ResponseOptimization;
}

export interface ResponseOptimization {
  streaming: boolean;
  caching: boolean;
  compression: boolean;
  batching: boolean;
  priorityQueue: boolean;
  maxBatchSize: number;
  timeoutMs: number;
}

export interface TrainingDataField {
  id: string;
  name: string;
  description: string;
  datasets: DatasetConfig[];
  weight: number;
  specializations: string[];
}

export interface DatasetConfig {
  id: string;
  name: string;
  tokenCount: number;
  sampleCount: number;
  quality: number;
  lastUpdated: Date;
  tags: string[];
  language: string[];
  domain: string[];
}

// Enhanced Model Configuration with 1M Context
export const ENHANCED_MODEL_CONFIG: AIModelConfig = {
  id: 'z-ai-enhanced-v2',
  name: 'Z.AI Enhanced Model v2',
  maxContextTokens: 1000000, // 1M tokens
  maxInputTokens: 128000, // 128K input tokens
  maxOutputTokens: 8192,
  temperature: 0.7,
  topP: 0.95,
  topK: 40,
  presencePenalty: 0.1,
  frequencyPenalty: 0.1,
  responseOptimization: {
    streaming: true,
    caching: true,
    compression: true,
    batching: true,
    priorityQueue: true,
    maxBatchSize: 32,
    timeoutMs: 30000 // 30 second timeout
  }
};

// Expanded Training Data Fields
export const EXPANDED_TRAINING_FIELDS: TrainingDataField[] = [
  {
    id: 'stem',
    name: 'STEM & Technology',
    description: 'Science, Technology, Engineering, Mathematics',
    datasets: [
      {
        id: 'arxiv-papers',
        name: 'ArXiv Research Papers',
        tokenCount: 50000000000, // 50B tokens
        sampleCount: 2500000,
        quality: 0.95,
        lastUpdated: new Date('2024-01-01'),
        tags: ['research', 'academic', 'peer-reviewed'],
        language: ['en'],
        domain: ['computer-science', 'mathematics', 'physics', 'biology']
      },
      {
        id: 'github-code',
        name: 'GitHub Code Repository',
        tokenCount: 30000000000, // 30B tokens
        sampleCount: 5000000,
        quality: 0.92,
        lastUpdated: new Date('2024-01-01'),
        tags: ['code', 'programming', 'open-source'],
        language: ['javascript', 'python', 'java', 'cpp', 'go', 'rust'],
        domain: ['software-development', 'algorithms', 'data-structures']
      },
      {
        id: 'stack-overflow',
        name: 'Stack Overflow Q&A',
        tokenCount: 15000000000, // 15B tokens
        sampleCount: 10000000,
        quality: 0.89,
        lastUpdated: new Date('2024-01-01'),
        tags: ['qa', 'programming', 'troubleshooting'],
        language: ['en'],
        domain: ['programming', 'software-engineering', 'debugging']
      }
    ],
    weight: 0.25,
    specializations: ['machine-learning', 'ai', 'data-science', 'robotics']
  },
  {
    id: 'humanities',
    name: 'Humanities & Arts',
    description: 'Literature, History, Philosophy, Arts',
    datasets: [
      {
        id: 'project-gutenberg',
        name: 'Project Gutenberg Library',
        tokenCount: 20000000000, // 20B tokens
        sampleCount: 75000,
        quality: 0.94,
        lastUpdated: new Date('2024-01-01'),
        tags: ['literature', 'classic', 'public-domain'],
        language: ['en', 'fr', 'de', 'es', 'it'],
        domain: ['literature', 'poetry', 'drama', 'philosophy']
      },
      {
        id: 'academic-journals',
        name: 'Academic Humanities Journals',
        tokenCount: 12000000000, // 12B tokens
        sampleCount: 500000,
        quality: 0.96,
        lastUpdated: new Date('2024-01-01'),
        tags: ['academic', 'peer-reviewed', 'research'],
        language: ['en'],
        domain: ['history', 'philosophy', 'linguistics', 'archaeology']
      }
    ],
    weight: 0.20,
    specializations: ['literature-analysis', 'historical-research', 'philosophical-reasoning']
  },
  {
    id: 'business',
    name: 'Business & Finance',
    description: 'Business, Finance, Economics, Management',
    datasets: [
      {
        id: 'financial-reports',
        name: 'Financial Reports & Analysis',
        tokenCount: 8000000000, // 8B tokens
        sampleCount: 2000000,
        quality: 0.91,
        lastUpdated: new Date('2024-01-01'),
        tags: ['finance', 'reports', 'analysis'],
        language: ['en'],
        domain: ['finance', 'accounting', 'investment', 'banking']
      },
      {
        id: 'business-case-studies',
        name: 'Harvard Business Case Studies',
        tokenCount: 5000000000, // 5B tokens
        sampleCount: 50000,
        quality: 0.93,
        lastUpdated: new Date('2024-01-01'),
        tags: ['business', 'case-studies', 'management'],
        language: ['en'],
        domain: ['business-strategy', 'management', 'marketing', 'entrepreneurship']
      }
    ],
    weight: 0.15,
    specializations: ['financial-analysis', 'business-strategy', 'market-research']
  },
  {
    id: 'healthcare',
    name: 'Healthcare & Medicine',
    description: 'Medical Research, Clinical Data, Healthcare',
    datasets: [
      {
        id: 'pubmed-abstracts',
        name: 'PubMed Medical Abstracts',
        tokenCount: 15000000000, // 15B tokens
        sampleCount: 30000000,
        quality: 0.97,
        lastUpdated: new Date('2024-01-01'),
        tags: ['medical', 'research', 'clinical'],
        language: ['en'],
        domain: ['medicine', 'healthcare', 'pharmaceuticals', 'biotechnology']
      },
      {
        id: 'clinical-trials',
        name: 'Clinical Trials Data',
        tokenCount: 7000000000, // 7B tokens
        sampleCount: 400000,
        quality: 0.95,
        lastUpdated: new Date('2024-01-01'),
        tags: ['clinical', 'trials', 'research'],
        language: ['en'],
        domain: ['clinical-research', 'drug-development', 'medical-devices']
      }
    ],
    weight: 0.15,
    specializations: ['medical-research', 'clinical-analysis', 'healthcare-ai']
  },
  {
    id: 'legal',
    name: 'Legal & Regulatory',
    description: 'Legal Documents, Case Law, Regulations',
    datasets: [
      {
        id: 'court-decisions',
        name: 'Court Decisions & Case Law',
        tokenCount: 10000000000, // 10B tokens
        sampleCount: 5000000,
        quality: 0.94,
        lastUpdated: new Date('2024-01-01'),
        tags: ['legal', 'court', 'decisions'],
        language: ['en'],
        domain: ['constitutional-law', 'corporate-law', 'intellectual-property']
      },
      {
        id: 'regulatory-documents',
        name: 'Government Regulations',
        tokenCount: 6000000000, // 6B tokens
        sampleCount: 1000000,
        quality: 0.92,
        lastUpdated: new Date('2024-01-01'),
        tags: ['regulatory', 'government', 'compliance'],
        language: ['en'],
        domain: ['regulatory-compliance', 'administrative-law', 'policy']
      }
    ],
    weight: 0.10,
    specializations: ['legal-research', 'compliance', 'regulatory-analysis']
  },
  {
    id: 'education',
    name: 'Education & Learning',
    description: 'Educational Content, Learning Materials',
    datasets: [
      {
        id: 'textbooks',
        name: 'Digital Textbooks',
        tokenCount: 8000000000, // 8B tokens
        sampleCount: 100000,
        quality: 0.90,
        lastUpdated: new Date('2024-01-01'),
        tags: ['education', 'textbooks', 'learning'],
        language: ['en'],
        domain: ['education', 'pedagogy', 'curriculum', 'assessment']
      },
      {
        id: 'mooc-content',
        name: 'MOOC Course Content',
        tokenCount: 5000000000, // 5B tokens
        sampleCount: 50000,
        quality: 0.88,
        lastUpdated: new Date('2024-01-01'),
        tags: ['online-learning', 'courses', 'mooc'],
        language: ['en'],
        domain: ['online-education', 'distance-learning', 'professional-development']
      }
    ],
    weight: 0.10,
    specializations: ['educational-content', 'learning-analytics', 'curriculum-design']
  },
  {
    id: 'multilingual',
    name: 'Multilingual Content',
    description: 'Content in Multiple Languages',
    datasets: [
      {
        id: 'wikipedia-multilingual',
        name: 'Wikipedia Multilingual',
        tokenCount: 20000000000, // 20B tokens
        sampleCount: 5000000,
        quality: 0.91,
        lastUpdated: new Date('2024-01-01'),
        tags: ['encyclopedia', 'multilingual', 'knowledge'],
        language: ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ru', 'ar'],
        domain: ['general-knowledge', 'encyclopedia', 'multilingual']
      },
      {
        id: 'common-crawl-multilingual',
        name: 'Common Crawl Multilingual',
        tokenCount: 50000000000, // 50B tokens
        sampleCount: 100000000,
        quality: 0.85,
        lastUpdated: new Date('2024-01-01'),
        tags: ['web-content', 'multilingual', 'diverse'],
        language: ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ru', 'ar', 'hi', 'pt'],
        domain: ['web-content', 'multilingual', 'diverse-topics']
      }
    ],
    weight: 0.05,
    specializations: ['translation', 'multilingual-understanding', 'cross-cultural']
  }
];

// Performance Optimization Settings
export const PERFORMANCE_SETTINGS = {
  responseTime: {
    target: 2000, // 2 seconds target
    max: 5000, // 5 seconds max
    optimization: {
      caching: true,
      compression: true,
      streaming: true,
      batching: true,
      preload: true
    }
  },
  memory: {
    maxContextSize: 1000000, // 1M tokens
    cacheSize: 500000, // 500K tokens cache
    compressionThreshold: 100000 // 100K tokens compression threshold
  },
  throughput: {
    maxConcurrentRequests: 100,
    batchSize: 32,
    queueTimeout: 30000
  }
};

const aiConfig = {
  ENHANCED_MODEL_CONFIG,
  EXPANDED_TRAINING_FIELDS,
  PERFORMANCE_SETTINGS
};

export default aiConfig;