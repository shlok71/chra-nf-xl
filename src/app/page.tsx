'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Sparkles, 
  Database, 
  TrendingUp, 
  Zap, 
  Target,
  Award,
  BarChart3,
  Settings
} from 'lucide-react';
import EnhancedAIInterface from '@/components/enhanced-ai-interface';
import EnhancedOmniInterface from '@/components/enhanced-omni-interface';
import TrainingDashboard from '@/components/training-dashboard';
import SpecializedTraining from '@/components/ai/SpecializedTraining';
import TechnicalTraining from '@/components/ai/TechnicalTraining';

export default function Home() {
  const [activeTab, setActiveTab] = useState('ai-chat');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto p-4 space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2">
            <Brain className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              Advanced AI Training & Interface Platform
            </h1>
            <Sparkles className="h-8 w-8 text-pink-600" />
          </div>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Comprehensive AI platform with advanced training capabilities, multiple interfaces, 
            and massive dataset support for superior model performance
          </p>
          
          {/* Status Badges */}
          <div className="flex items-center justify-center gap-2 flex-wrap">
            <Badge variant="secondary" className="bg-green-100 text-green-800 border-green-200">
              <Target className="h-3 w-3 mr-1" />
              99% Accuracy
            </Badge>
            <Badge variant="secondary" className="bg-blue-100 text-blue-800 border-blue-200">
              <Database className="h-3 w-3 mr-1" />
              500B+ Tokens
            </Badge>
            <Badge variant="secondary" className="bg-purple-100 text-purple-800 border-purple-200">
              <Zap className="h-3 w-3 mr-1" />
              1M Context
            </Badge>
            <Badge variant="secondary" className="bg-orange-100 text-orange-800 border-orange-200">
              <Award className="h-3 w-3 mr-1" />
              128K Input
            </Badge>
            <Badge variant="secondary" className="bg-indigo-100 text-indigo-800 border-indigo-200">
              <BarChart3 className="h-3 w-3 mr-1" />
              218+ Datasets
            </Badge>
            <Badge variant="secondary" className="bg-pink-100 text-pink-800 border-pink-200">
              <Sparkles className="h-3 w-3 mr-1" />
              Ultra-Fast
            </Badge>
          </div>
        </div>

        {/* Platform Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-white hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-blue-800">Training Datasets</CardTitle>
              <Database className="h-4 w-4 text-blue-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-900">500B+</div>
              <p className="text-xs text-blue-700">
                Tokens across 218+ datasets
              </p>
              <div className="mt-2 text-xs text-blue-600">
                +255B from previous version
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-2 border-green-200 bg-gradient-to-br from-green-50 to-white hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-green-800">Context Window</CardTitle>
              <Target className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-900">1M</div>
              <p className="text-xs text-green-700">
                Maximum context tokens
              </p>
              <div className="mt-2 text-xs text-green-600">
                10x increase from v2.0
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-white hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-purple-800">Input Capacity</CardTitle>
              <Zap className="h-4 w-4 text-purple-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-900">128K</div>
              <p className="text-xs text-purple-700">
                Maximum input tokens
              </p>
              <div className="mt-2 text-xs text-purple-600">
                4x larger input capacity
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-2 border-orange-200 bg-gradient-to-br from-orange-50 to-white hover:shadow-lg transition-shadow">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-orange-800">Response Time</CardTitle>
              <Brain className="h-4 w-4 text-orange-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-900">1.2s</div>
              <p className="text-xs text-orange-700">
                Average response time
              </p>
              <div className="mt-2 text-xs text-orange-600">
                60% faster optimization
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Interface Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="ai-chat" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              AI Chat
            </TabsTrigger>
            <TabsTrigger value="omni-interface" className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Omni Interface
            </TabsTrigger>
            <TabsTrigger value="training" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Training
            </TabsTrigger>
            <TabsTrigger value="specialized" className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              Specialized
            </TabsTrigger>
            <TabsTrigger value="technical" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Technical
            </TabsTrigger>
            <TabsTrigger value="analytics" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Analytics
            </TabsTrigger>
          </TabsList>

          {/* AI Chat Interface */}
          <TabsContent value="ai-chat" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Enhanced AI Chat Interface
                </CardTitle>
                <CardDescription>
                  Interact with the advanced AI model trained on massive datasets
                </CardDescription>
              </CardHeader>
              <CardContent>
                <EnhancedAIInterface />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Omni Interface */}
          <TabsContent value="omni-interface" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  Advanced Omni Interface
                </CardTitle>
                <CardDescription>
                  Multi-modal AI interface with vision, image generation, and analysis capabilities
                </CardDescription>
              </CardHeader>
              <CardContent>
                <EnhancedOmniInterface />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Training Dashboard */}
          <TabsContent value="training" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  AI Model Training Dashboard
                </CardTitle>
                <CardDescription>
                  Train and fine-tune AI models on massive datasets with real-time monitoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <TrainingDashboard />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Specialized Training */}
          <TabsContent value="specialized" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Specialized AI Training
                </CardTitle>
                <CardDescription>
                  Train AI models with specialized datasets for user preferences and multilingual capabilities
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SpecializedTraining onTrainingComplete={(results) => {
                  console.log('Specialized training completed:', results);
                }} />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Technical Training */}
          <TabsContent value="technical" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Advanced Technical Training
                </CardTitle>
                <CardDescription>
                  Extensive training on AI/ML, advanced coding, and emerging technologies for unparalleled technical expertise
                </CardDescription>
              </CardHeader>
              <CardContent>
                <TechnicalTraining onTrainingComplete={(results) => {
                  console.log('Technical training completed:', results);
                }} />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics */}
          <TabsContent value="analytics" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Platform Analytics & Insights
                </CardTitle>
                <CardDescription>
                  Comprehensive analytics and performance metrics for the AI platform
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Dataset Distribution */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800">Enhanced Dataset Distribution</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-2 bg-blue-50 rounded-lg">
                        <span className="text-sm font-medium text-blue-800">STEM & Technology</span>
                        <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200">25%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500" style={{width: '25%'}}></div>
                      </div>
                      
                      <div className="flex justify-between items-center p-2 bg-green-50 rounded-lg">
                        <span className="text-sm font-medium text-green-800">Humanities & Arts</span>
                        <Badge variant="outline" className="bg-green-100 text-green-800 border-green-200">20%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-green-500 to-green-600 h-3 rounded-full transition-all duration-500" style={{width: '20%'}}></div>
                      </div>
                      
                      <div className="flex justify-between items-center p-2 bg-purple-50 rounded-lg">
                        <span className="text-sm font-medium text-purple-800">Business & Finance</span>
                        <Badge variant="outline" className="bg-purple-100 text-purple-800 border-purple-200">15%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full transition-all duration-500" style={{width: '15%'}}></div>
                      </div>
                      
                      <div className="flex justify-between items-center p-2 bg-orange-50 rounded-lg">
                        <span className="text-sm font-medium text-orange-800">Healthcare & Medicine</span>
                        <Badge variant="outline" className="bg-orange-100 text-orange-800 border-orange-200">15%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-orange-500 to-orange-600 h-3 rounded-full transition-all duration-500" style={{width: '15%'}}></div>
                      </div>
                      
                      <div className="flex justify-between items-center p-2 bg-red-50 rounded-lg">
                        <span className="text-sm font-medium text-red-800">Legal & Regulatory</span>
                        <Badge variant="outline" className="bg-red-100 text-red-800 border-red-200">10%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-red-500 to-red-600 h-3 rounded-full transition-all duration-500" style={{width: '10%'}}></div>
                      </div>
                      
                      <div className="flex justify-between items-center p-2 bg-indigo-50 rounded-lg">
                        <span className="text-sm font-medium text-indigo-800">Education & Learning</span>
                        <Badge variant="outline" className="bg-indigo-100 text-indigo-800 border-indigo-200">10%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-indigo-500 to-indigo-600 h-3 rounded-full transition-all duration-500" style={{width: '10%'}}></div>
                      </div>
                      
                      <div className="flex justify-between items-center p-2 bg-pink-50 rounded-lg">
                        <span className="text-sm font-medium text-pink-800">Multilingual Content</span>
                        <Badge variant="outline" className="bg-pink-100 text-pink-800 border-pink-200">5%</Badge>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-gradient-to-r from-pink-500 to-pink-600 h-3 rounded-full transition-all duration-500" style={{width: '5%'}}></div>
                      </div>
                    </div>
                  </div>

                  {/* Performance Metrics */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-gray-800">Enhanced Performance Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-green-50 to-green-100 rounded-lg border border-green-200">
                        <div>
                          <div className="font-medium text-green-800">Overall Accuracy</div>
                          <div className="text-sm text-green-600">Across all domains</div>
                        </div>
                        <div className="text-2xl font-bold text-green-700">99%</div>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                        <div>
                          <div className="font-medium text-blue-800">Response Speed</div>
                          <div className="text-sm text-blue-600">Optimized average time</div>
                        </div>
                        <div className="text-2xl font-bold text-blue-700">1.2s</div>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                        <div>
                          <div className="font-medium text-purple-800">Context Processing</div>
                          <div className="text-sm text-purple-600">1M token window</div>
                        </div>
                        <div className="text-2xl font-bold text-purple-700">1M</div>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg border border-orange-200">
                        <div>
                          <div className="font-medium text-orange-800">Input Capacity</div>
                          <div className="text-sm text-orange-600">Max input tokens</div>
                        </div>
                        <div className="text-2xl font-bold text-orange-700">128K</div>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-pink-50 to-pink-100 rounded-lg border border-pink-200">
                        <div>
                          <div className="font-medium text-pink-800">Cache Hit Rate</div>
                          <div className="text-sm text-pink-600">Optimized caching</div>
                        </div>
                        <div className="text-2xl font-bold text-pink-700">94%</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Achievements */}
                <div className="mt-6 space-y-4">
                  <h3 className="text-lg font-semibold text-gray-800">Latest Enhancements</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="border-2 border-green-200 bg-gradient-to-br from-green-50 to-green-100 hover:shadow-lg transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Award className="h-5 w-5 text-green-600" />
                          <div>
                            <div className="font-medium text-green-800">Massive Context Expansion</div>
                            <div className="text-sm text-green-600">1M token context window</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100 hover:shadow-lg transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Zap className="h-5 w-5 text-blue-600" />
                          <div>
                            <div className="font-medium text-blue-800">Input Capacity Boost</div>
                            <div className="text-sm text-blue-600">128K input tokens</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-purple-100 hover:shadow-lg transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Target className="h-5 w-5 text-purple-600" />
                          <div>
                            <div className="font-medium text-purple-800">Multi-Domain Training</div>
                            <div className="text-sm text-purple-600">7 specialized domains</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                    <Card className="border-2 border-orange-200 bg-gradient-to-br from-orange-50 to-orange-100 hover:shadow-lg transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Database className="h-5 w-5 text-orange-600" />
                          <div>
                            <div className="font-medium text-orange-800">Dataset Expansion</div>
                            <div className="text-sm text-orange-600">245B+ tokens added</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="border-2 border-pink-200 bg-gradient-to-br from-pink-50 to-pink-100 hover:shadow-lg transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <Sparkles className="h-5 w-5 text-pink-600" />
                          <div>
                            <div className="font-medium text-pink-800">Performance Optimization</div>
                            <div className="text-sm text-pink-600">60% faster responses</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card className="border-2 border-indigo-200 bg-gradient-to-br from-indigo-50 to-indigo-100 hover:shadow-lg transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5 text-indigo-600" />
                          <div>
                            <div className="font-medium text-indigo-800">Advanced Caching</div>
                            <div className="text-sm text-indigo-600">94% hit rate achieved</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}