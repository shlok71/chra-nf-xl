'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Heart, 
  Languages, 
  Brain, 
  Target, 
  CheckCircle2, 
  Clock,
  TrendingUp,
  Users,
  Globe,
  MessageSquare,
  Sparkles
} from 'lucide-react';
import TrainingSummary from './TrainingSummary';

interface TrainingResult {
  name: string;
  description: string;
  size: string;
  focus?: string;
  languages?: number;
  status: string;
  accuracy: string;
  trainingTime: string;
  samples: string;
}

interface SpecializedTrainingProps {
  onTrainingComplete: (results: any) => void;
}

export default function SpecializedTraining({ onTrainingComplete }: SpecializedTrainingProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingType, setTrainingType] = useState<'user-preference' | 'multilingual' | null>(null);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<{
    userPreference?: TrainingResult[];
    multilingual?: TrainingResult[];
    improvements?: any;
    trainingComplete?: any;
  }>({});

  const startTraining = async (type: 'user-preference' | 'multilingual') => {
    setIsTraining(true);
    setTrainingType(type);
    setProgress(0);

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval);
          return 95;
        }
        return prev + Math.random() * 15;
      });
    }, 500);

    try {
      const response = await fetch('/api/ai/train-specialized', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trainingType: type })
      });

      const data = await response.json();
      
      if (data.success) {
        setProgress(100);
        setResults(prev => ({
          ...prev,
          [type === 'user-preference' ? 'userPreference' : 'multilingual']: data.results,
          improvements: data.improvements,
          trainingComplete: data
        }));
        
        setTimeout(() => {
          setIsTraining(false);
          onTrainingComplete(data);
        }, 1000);
      }
    } catch (error) {
      console.error('Training failed:', error);
      setIsTraining(false);
    }

    clearInterval(progressInterval);
  };

  const handleExportResults = () => {
    const dataStr = JSON.stringify(results.trainingComplete, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `training-results-${trainingType}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const handleShareResults = () => {
    if (navigator.share) {
      navigator.share({
        title: 'AI Training Results',
        text: `Completed ${trainingType} training with ${results.trainingComplete?.overallImprovement}% overall improvement`,
        url: window.location.href
      });
    } else {
      // Fallback - copy to clipboard
      navigator.clipboard.writeText(
        `AI Training Results - ${trainingType}: ${results.trainingComplete?.overallImprovement}% improvement`
      );
    }
  };

  // Show training summary if training is complete
  if (results.trainingComplete) {
    return (
      <TrainingSummary 
        trainingResults={results.trainingComplete}
        onExport={handleExportResults}
        onShare={handleShareResults}
      />
    );
  }

  const UserPreferenceMetrics = () => (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      <div className="text-center p-4 bg-gradient-to-br from-pink-50 to-purple-50 rounded-lg border border-pink-200">
        <Heart className="h-8 w-8 mx-auto mb-2 text-pink-600" />
        <div className="text-2xl font-bold text-pink-700">
          {results.improvements?.responseSatisfaction || '--'}
        </div>
        <div className="text-sm text-gray-600">Response Satisfaction</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
        <MessageSquare className="h-8 w-8 mx-auto mb-2 text-blue-600" />
        <div className="text-2xl font-bold text-blue-700">
          {results.improvements?.conversationFlow || '--'}
        </div>
        <div className="text-sm text-gray-600">Conversation Flow</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg border border-green-200">
        <Users className="h-8 w-8 mx-auto mb-2 text-green-600" />
        <div className="text-2xl font-bold text-green-700">
          {results.improvements?.personalization || '--'}
        </div>
        <div className="text-sm text-gray-600">Personalization</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-yellow-50 to-amber-50 rounded-lg border border-yellow-200">
        <Sparkles className="h-8 w-8 mx-auto mb-2 text-yellow-600" />
        <div className="text-2xl font-bold text-yellow-700">
          {results.improvements?.clarityScore || '--'}
        </div>
        <div className="text-sm text-gray-600">Clarity Score</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200">
        <Target className="h-8 w-8 mx-auto mb-2 text-purple-600" />
        <div className="text-2xl font-bold text-purple-700">
          {results.improvements?.helpfulness || '--'}
        </div>
        <div className="text-sm text-gray-600">Helpfulness</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-indigo-50 to-blue-50 rounded-lg border border-indigo-200">
        <Brain className="h-8 w-8 mx-auto mb-2 text-indigo-600" />
        <div className="text-2xl font-bold text-indigo-700">
          {results.improvements?.adaptability || '--'}
        </div>
        <div className="text-sm text-gray-600">Adaptability</div>
      </div>
    </div>
  );

  const MultilingualMetrics = () => (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
        <Languages className="h-8 w-8 mx-auto mb-2 text-blue-600" />
        <div className="text-2xl font-bold text-blue-700">
          {results.improvements?.languageAccuracy || '--'}
        </div>
        <div className="text-sm text-gray-600">Language Accuracy</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border border-purple-200">
        <Globe className="h-8 w-8 mx-auto mb-2 text-purple-600" />
        <div className="text-2xl font-bold text-purple-700">
          {results.improvements?.culturalUnderstanding || '--'}
        </div>
        <div className="text-sm text-gray-600">Cultural Understanding</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg border border-green-200">
        <MessageSquare className="h-8 w-8 mx-auto mb-2 text-green-600" />
        <div className="text-2xl font-bold text-green-700">
          {results.improvements?.dialectProcessing || '--'}
        </div>
        <div className="text-sm text-gray-600">Dialect Processing</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-yellow-50 to-amber-50 rounded-lg border border-yellow-200">
        <TrendingUp className="h-8 w-8 mx-auto mb-2 text-yellow-600" />
        <div className="text-2xl font-bold text-yellow-700">
          {results.improvements?.crossLingualTransfer || '--'}
        </div>
        <div className="text-sm text-gray-600">Cross-Lingual Transfer</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-indigo-50 to-blue-50 rounded-lg border border-indigo-200">
        <Users className="h-8 w-8 mx-auto mb-2 text-indigo-600" />
        <div className="text-2xl font-bold text-indigo-700">
          {results.improvements?.regionalAdaptation || '--'}
        </div>
        <div className="text-sm text-gray-600">Regional Adaptation</div>
      </div>
      <div className="text-center p-4 bg-gradient-to-br from-pink-50 to-rose-50 rounded-lg border border-pink-200">
        <Brain className="h-8 w-8 mx-auto mb-2 text-pink-600" />
        <div className="text-2xl font-bold text-pink-700">
          {results.improvements?.technicalMultilingual || '--'}
        </div>
        <div className="text-sm text-gray-600">Technical Multilingual</div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Specialized AI Training
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Train your AI model with specialized datasets for user preferences and multilingual capabilities
        </p>
      </div>

      <Tabs defaultValue="user-preference" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="user-preference" className="flex items-center gap-2">
            <Heart className="h-4 w-4" />
            User Preferences
          </TabsTrigger>
          <TabsTrigger value="multilingual" className="flex items-center gap-2">
            <Languages className="h-4 w-4" />
            Multilingual
          </TabsTrigger>
        </TabsList>

        <TabsContent value="user-preference" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Heart className="h-5 w-5 text-pink-600" />
                User Response Preference Training
              </CardTitle>
              <CardDescription>
                Train the AI to understand and adapt to user response preferences, communication styles, and satisfaction patterns
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-pink-50 rounded-lg border border-pink-200">
                  <h4 className="font-semibold text-pink-800 mb-2">Training Focus</h4>
                  <ul className="text-sm text-pink-700 space-y-1">
                    <li>• Response style adaptation</li>
                    <li>• Tone and format preferences</li>
                    <li>• Conversation flow optimization</li>
                    <li>• Personalization patterns</li>
                  </ul>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <h4 className="font-semibold text-purple-800 mb-2">Expected Improvements</h4>
                  <ul className="text-sm text-purple-700 space-y-1">
                    <li>• 85-95% satisfaction rate</li>
                    <li>• Enhanced personalization</li>
                    <li>• Better conversation flow</li>
                    <li>• Improved clarity and helpfulness</li>
                  </ul>
                </div>
              </div>

              {isTraining && trainingType === 'user-preference' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Training Progress</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                  <p className="text-sm text-gray-600 text-center">
                    Training on user preference datasets...
                  </p>
                </div>
              )}

              <Button 
                onClick={() => startTraining('user-preference')}
                disabled={isTraining}
                className="w-full bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700"
              >
                {isTraining && trainingType === 'user-preference' ? (
                  <>
                    <Clock className="mr-2 h-4 w-4 animate-spin" />
                    Training User Preferences...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Start User Preference Training
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {results.userPreference && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  Training Results - User Preferences
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <UserPreferenceMetrics />
                
                <div className="space-y-3">
                  <h4 className="font-semibold">Trained Datasets:</h4>
                  {results.userPreference.map((dataset, index) => (
                    <div key={index} className="p-3 bg-gray-50 rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h5 className="font-medium">{dataset.name}</h5>
                          <p className="text-sm text-gray-600">{dataset.description}</p>
                        </div>
                        <Badge variant="secondary">{dataset.accuracy}</Badge>
                      </div>
                      <div className="flex gap-4 text-sm text-gray-500">
                        <span>Size: {dataset.size}</span>
                        <span>Time: {dataset.trainingTime}</span>
                        <span>Samples: {dataset.samples}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="multilingual" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Languages className="h-5 w-5 text-blue-600" />
                Multilingual Capability Training
              </CardTitle>
              <CardDescription>
                Enhance the AI's multilingual capabilities with comprehensive language and cultural context training
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <h4 className="font-semibold text-blue-800 mb-2">Training Focus</h4>
                  <ul className="text-sm text-blue-700 space-y-1">
                    <li>• 50+ language support</li>
                    <li>• Cultural context understanding</li>
                    <li>• Regional dialect processing</li>
                    <li>• Cross-lingual transfer</li>
                  </ul>
                </div>
                <div className="p-4 bg-cyan-50 rounded-lg border border-cyan-200">
                  <h4 className="font-semibold text-cyan-800 mb-2">Expected Improvements</h4>
                  <ul className="text-sm text-cyan-700 space-y-1">
                    <li>• 80-95% language accuracy</li>
                    <li>• Enhanced cultural awareness</li>
                    <li>• Better dialect handling</li>
                    <li>• Improved technical translations</li>
                  </ul>
                </div>
              </div>

              {isTraining && trainingType === 'multilingual' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Training Progress</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                  <p className="text-sm text-gray-600 text-center">
                    Training on multilingual datasets...
                  </p>
                </div>
              )}

              <Button 
                onClick={() => startTraining('multilingual')}
                disabled={isTraining}
                className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
              >
                {isTraining && trainingType === 'multilingual' ? (
                  <>
                    <Clock className="mr-2 h-4 w-4 animate-spin" />
                    Training Multilingual Capabilities...
                  </>
                ) : (
                  <>
                    <Globe className="mr-2 h-4 w-4" />
                    Start Multilingual Training
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {results.multilingual && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  Training Results - Multilingual
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <MultilingualMetrics />
                
                <div className="space-y-3">
                  <h4 className="font-semibold">Trained Datasets:</h4>
                  {results.multilingual.map((dataset, index) => (
                    <div key={index} className="p-3 bg-gray-50 rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h5 className="font-medium">{dataset.name}</h5>
                          <p className="text-sm text-gray-600">{dataset.description}</p>
                          {dataset.languages && (
                            <p className="text-xs text-blue-600 mt-1">
                              Supports {dataset.languages} languages
                            </p>
                          )}
                        </div>
                        <Badge variant="secondary">{dataset.accuracy}</Badge>
                      </div>
                      <div className="flex gap-4 text-sm text-gray-500">
                        <span>Size: {dataset.size}</span>
                        <span>Time: {dataset.trainingTime}</span>
                        <span>Samples: {dataset.samples}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}