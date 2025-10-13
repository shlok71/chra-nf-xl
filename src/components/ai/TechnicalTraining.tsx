'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Brain, 
  Code, 
  Cpu, 
  Database, 
  CheckCircle2, 
  Clock,
  TrendingUp,
  Terminal,
  GitBranch,
  Layers,
  Zap,
  Award,
  Rocket,
  Lightbulb,
  Shield,
  Globe,
  Smartphone,
  Cloud,
  Lock,
  Atom
} from 'lucide-react';
import TechnicalTrainingSummary from './TechnicalTrainingSummary';

interface TechnicalDataset {
  name: string;
  description: string;
  size: string;
  focus: string;
  difficulty: string;
  languages: string[];
  frameworks: string[];
  status: string;
  accuracy: string;
  trainingTime: string;
  samples: string;
  codeQuality: string;
  problemSolving: string;
  technicalDepth: string;
}

interface TechnicalTrainingProps {
  onTrainingComplete: (results: any) => void;
}

export default function TechnicalTraining({ onTrainingComplete }: TechnicalTrainingProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingDomain, setTrainingDomain] = useState<'ai-ml' | 'coding' | 'emerging-tech' | null>(null);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<{
    aiMl?: TechnicalDataset[];
    coding?: TechnicalDataset[];
    emergingTech?: TechnicalDataset[];
    improvements?: any;
    trainingComplete?: any;
  }>({});

  const startTraining = async (domain: 'ai-ml' | 'coding' | 'emerging-tech') => {
    setIsTraining(true);
    setTrainingDomain(domain);
    setProgress(0);

    // Simulate progress updates (longer for technical training)
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return 95;
        }
        return prev + Math.random() * 10; // Slower progress for technical training
      });
    }, 800);

    try {
      const response = await fetch('/api/ai/train-technical', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trainingDomain: domain })
      });

      const data = await response.json();
      
      if (data.success) {
        setProgress(100);
        setResults(prev => ({
          ...prev,
          [domain === 'ai-ml' ? 'aiMl' : domain === 'coding' ? 'coding' : 'emergingTech']: data.results,
          improvements: data.improvements,
          trainingComplete: data
        }));
        
        setTimeout(() => {
          setIsTraining(false);
          onTrainingComplete(data);
        }, 1500);
      }
    } catch (error) {
      console.error('Technical training failed:', error);
      setIsTraining(false);
    }

    clearInterval(progressInterval);
  };

  const handleExportResults = () => {
    const dataStr = JSON.stringify(results.trainingComplete, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `technical-training-${trainingDomain}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleShareResults = () => {
    if (navigator.share) {
      navigator.share({
        title: 'Technical AI Training Results',
        text: `Completed ${trainingDomain} training with ${results.trainingComplete?.overallImprovement}% overall improvement`,
        url: window.location.href
      });
    } else {
      navigator.clipboard.writeText(
        `Technical AI Training - ${trainingDomain}: ${results.trainingComplete?.overallImprovement}% improvement`
      );
    }
  };

  // Show training summary if training is complete
  if (results.trainingComplete) {
    return (
      <TechnicalTrainingSummary 
        trainingResults={results.trainingComplete}
        onExport={handleExportResults}
        onShare={handleShareResults}
      />
    );
  }

  const getDomainIcon = (domain: string) => {
    switch (domain) {
      case 'ai-ml': return <Brain className="h-5 w-5" />;
      case 'coding': return <Code className="h-5 w-5" />;
      case 'emerging-tech': return <Rocket className="h-5 w-5" />;
      default: return <Cpu className="h-5 w-5" />;
    }
  };

  const getDomainColor = (domain: string) => {
    switch (domain) {
      case 'ai-ml': return 'from-blue-600 to-purple-600';
      case 'coding': return 'from-green-600 to-emerald-600';
      case 'emerging-tech': return 'from-orange-600 to-red-600';
      default: return 'from-gray-600 to-gray-700';
    }
  };

  const AIMLContent = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-600" />
            AI & Machine Learning Training
          </CardTitle>
          <CardDescription>
            Master advanced AI/ML concepts including deep learning, NLP, computer vision, and reinforcement learning
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">Training Focus</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Deep Learning & Neural Networks</li>
                <li>• Natural Language Processing</li>
                <li>• Computer Vision & Image Processing</li>
                <li>• Reinforcement Learning Systems</li>
                <li>• ML Model Optimization</li>
                <li>• Research Paper Implementation</li>
              </ul>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
              <h4 className="font-semibold text-purple-800 mb-2">Technical Stack</h4>
              <div className="space-y-2 text-sm text-purple-700">
                <div><strong>Languages:</strong> Python, R, Julia, C++</div>
                <div><strong>Frameworks:</strong> TensorFlow, PyTorch, Scikit-learn</div>
                <div><strong>Libraries:</strong> Hugging Face, OpenCV, NLTK</div>
                <div><strong>Tools:</strong> Jupyter, Colab, MLflow</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
              <Brain className="h-8 w-8 mx-auto mb-2 text-blue-600" />
              <div className="text-2xl font-bold text-blue-700">78B</div>
              <div className="text-sm text-gray-600">Training Tokens</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200">
              <Layers className="h-8 w-8 mx-auto mb-2 text-purple-600" />
              <div className="text-2xl font-bold text-purple-700">4</div>
              <div className="text-sm text-gray-600">Core Datasets</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg border border-green-200">
              <Award className="h-8 w-8 mx-auto mb-2 text-green-600" />
              <div className="text-2xl font-bold text-green-700">Expert</div>
              <div className="text-sm text-gray-600">Difficulty Level</div>
            </div>
          </div>

          {isTraining && trainingDomain === 'ai-ml' && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Training Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-gray-600 text-center">
                Training on advanced AI/ML datasets...
              </p>
            </div>
          )}

          <Button 
            onClick={() => startTraining('ai-ml')}
            disabled={isTraining}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
          >
            {isTraining && trainingDomain === 'ai-ml' ? (
              <>
                <Clock className="mr-2 h-4 w-4 animate-spin" />
                Training AI/ML Expertise...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                Start AI/ML Training
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );

  const CodingContent = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5 text-green-600" />
            Advanced Coding & Development
          </CardTitle>
          <CardDescription>
            Comprehensive training in full-stack development, system programming, mobile apps, and DevOps
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
              <h4 className="font-semibold text-green-800 mb-2">Development Areas</h4>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Full-Stack Web Development</li>
                <li>• System Programming & Architecture</li>
                <li>• Mobile App Development</li>
                <li>• DevOps & Cloud Infrastructure</li>
                <li>• API Design & Development</li>
                <li>• Database Management</li>
              </ul>
            </div>
            <div className="p-4 bg-emerald-50 rounded-lg border border-emerald-200">
              <h4 className="font-semibold text-emerald-800 mb-2">Technology Stack</h4>
              <div className="space-y-2 text-sm text-emerald-700">
                <div><strong>Languages:</strong> JavaScript, TypeScript, Python, Go, Rust</div>
                <div><strong>Frameworks:</strong> React, Next.js, Node.js, Django</div>
                <div><strong>Cloud:</strong> AWS, Azure, GCP, Docker, K8s</div>
                <div><strong>Mobile:</strong> React Native, Flutter, Swift, Kotlin</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg border border-green-200">
              <Terminal className="h-8 w-8 mx-auto mb-2 text-green-600" />
              <div className="text-2xl font-bold text-green-700">76B</div>
              <div className="text-sm text-gray-600">Training Tokens</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
              <GitBranch className="h-8 w-8 mx-auto mb-2 text-blue-600" />
              <div className="text-2xl font-bold text-blue-700">4</div>
              <div className="text-sm text-gray-600">Core Domains</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200">
              <Zap className="h-8 w-8 mx-auto mb-2 text-purple-600" />
              <div className="text-2xl font-bold text-purple-700">Advanced</div>
              <div className="text-sm text-gray-600">Skill Level</div>
            </div>
          </div>

          {isTraining && trainingDomain === 'coding' && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Training Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-gray-600 text-center">
                Training on advanced coding datasets...
              </p>
            </div>
          )}

          <Button 
            onClick={() => startTraining('coding')}
            disabled={isTraining}
            className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
          >
            {isTraining && trainingDomain === 'coding' ? (
              <>
                <Clock className="mr-2 h-4 w-4 animate-spin" />
                Training Coding Expertise...
              </>
            ) : (
              <>
                <Code className="mr-2 h-4 w-4" />
                Start Coding Training
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );

  const EmergingTechContent = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Rocket className="h-5 w-5 text-orange-600" />
            Emerging Technologies
          </CardTitle>
          <CardDescription>
            Cutting-edge training in quantum computing, blockchain, IoT, edge computing, and cybersecurity
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
              <h4 className="font-semibold text-orange-800 mb-2">Frontier Technologies</h4>
              <ul className="text-sm text-orange-700 space-y-1">
                <li>• Quantum Computing & Algorithms</li>
                <li>• Blockchain & Web3 Development</li>
                <li>• Edge Computing & IoT</li>
                <li>• Cybersecurity & Ethical Hacking</li>
                <li>• Augmented & Virtual Reality</li>
                <li>• Autonomous Systems</li>
              </ul>
            </div>
            <div className="p-4 bg-red-50 rounded-lg border border-red-200">
              <h4 className="font-semibold text-red-800 mb-2">Advanced Tools</h4>
              <div className="space-y-2 text-sm text-red-700">
                <div><strong>Quantum:</strong> Q#, Qiskit, Cirq, IBM Quantum</div>
                <div><strong>Blockchain:</strong> Solidity, Rust, Web3.js</div>
                <div><strong>IoT:</strong> Arduino, Raspberry Pi, AWS IoT</div>
                <div><strong>Security:</strong> Kali Linux, Metasploit, Wireshark</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-red-50 rounded-lg border border-orange-200">
              <Atom className="h-8 w-8 mx-auto mb-2 text-orange-600" />
              <div className="text-2xl font-bold text-orange-700">57B</div>
              <div className="text-sm text-gray-600">Training Tokens</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg border border-purple-200">
              <Lightbulb className="h-8 w-8 mx-auto mb-2 text-purple-600" />
              <div className="text-2xl font-bold text-purple-700">4</div>
              <div className="text-sm text-gray-600">Tech Domains</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-yellow-50 to-amber-50 rounded-lg border border-yellow-200">
              <Shield className="h-8 w-8 mx-auto mb-2 text-yellow-600" />
              <div className="text-2xl font-bold text-yellow-700">Expert</div>
              <div className="text-sm text-gray-600">Complexity</div>
            </div>
          </div>

          {isTraining && trainingDomain === 'emerging-tech' && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Training Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
              <p className="text-sm text-gray-600 text-center">
                Training on emerging technologies...
              </p>
            </div>
          )}

          <Button 
            onClick={() => startTraining('emerging-tech')}
            disabled={isTraining}
            className="w-full bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700"
          >
            {isTraining && trainingDomain === 'emerging-tech' ? (
              <>
                <Clock className="mr-2 h-4 w-4 animate-spin" />
                Training Emerging Tech...
              </>
            ) : (
              <>
                <Rocket className="mr-2 h-4 w-4" />
                Start Emerging Tech Training
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-green-600 via-blue-600 to-purple-600 bg-clip-text text-transparent">
          Advanced Technical Training
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Extensive training on AI/ML, advanced coding, and emerging technologies for unparalleled technical expertise
        </p>
      </div>

      <Tabs defaultValue="ai-ml" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="ai-ml" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            AI & ML
          </TabsTrigger>
          <TabsTrigger value="coding" className="flex items-center gap-2">
            <Code className="h-4 w-4" />
            Coding
          </TabsTrigger>
          <TabsTrigger value="emerging-tech" className="flex items-center gap-2">
            <Rocket className="h-4 w-4" />
            Emerging Tech
          </TabsTrigger>
        </TabsList>

        <TabsContent value="ai-ml">
          <AIMLContent />
        </TabsContent>

        <TabsContent value="coding">
          <CodingContent />
        </TabsContent>

        <TabsContent value="emerging-tech">
          <EmergingTechContent />
        </TabsContent>
      </Tabs>
    </div>
  );
}