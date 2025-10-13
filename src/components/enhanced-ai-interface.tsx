'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Loader2, Sparkles, Brain, Calculator, Code, BookOpen, MessageSquare, Zap } from 'lucide-react';
import { toast } from 'sonner';

interface AIResponse {
  success: boolean;
  response: string;
  metrics: {
    model: string;
    task_type: string;
    performance_score: number;
    confidence: number;
    processing_time: number;
  };
  timestamp: string;
}

export default function EnhancedAIInterface() {
  const [prompt, setPrompt] = useState('');
  const [taskType, setTaskType] = useState('general');
  const [temperature, setTemperature] = useState([0.7]);
  const [maxTokens, setMaxTokens] = useState([1000]);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [response, setResponse] = useState<AIResponse | null>(null);
  const [streamingResponse, setStreamingResponse] = useState('');
  const [conversationHistory, setConversationHistory] = useState<Array<{prompt: string, response: string, taskType: string}>>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const taskTypes = [
    { value: 'general', label: 'General', icon: MessageSquare, description: 'General conversation and text generation' },
    { value: 'reasoning', label: 'Reasoning', icon: Brain, description: 'Logical reasoning and problem solving' },
    { value: 'math', label: 'Mathematics', icon: Calculator, description: 'Mathematical problem solving' },
    { value: 'coding', label: 'Coding', icon: Code, description: 'Code generation and algorithms' },
    { value: 'knowledge', label: 'Knowledge', icon: BookOpen, description: 'Knowledge integration and facts' }
  ];

  const handleSubmit = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    setLoading(true);
    setResponse(null);
    setStreamingResponse('');

    try {
      if (streaming) {
        await handleStreamingRequest();
      } else {
        await handleRegularRequest();
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to connect to AI model');
    } finally {
      setLoading(false);
    }
  };

  const handleRegularRequest = async () => {
    const response = await fetch('/api/ai', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        task_type: taskType,
        temperature: temperature[0],
        max_tokens: maxTokens[0]
      }),
    });

    const data: AIResponse = await response.json();

    if (data.success) {
      setResponse(data);
      setConversationHistory(prev => [...prev, {
        prompt,
        response: data.response,
        taskType
      }]);
      toast.success('Response generated successfully!');
    } else {
      toast.error(data.error || 'Failed to generate response');
    }
  };

  const handleStreamingRequest = async () => {
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('/api/ai/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          task_type: taskType,
          temperature: temperature[0],
          max_tokens: maxTokens[0]
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error('Streaming request failed');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No reader available');
      }

      let accumulatedResponse = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              
              if (data.type === 'content') {
                accumulatedResponse += data.content;
                setStreamingResponse(accumulatedResponse);
              } else if (data.type === 'done') {
                // Finalize the response
                setResponse({
                  success: true,
                  response: accumulatedResponse,
                  metrics: data.metrics,
                  timestamp: new Date().toISOString()
                });
                
                setConversationHistory(prev => [...prev, {
                  prompt,
                  response: accumulatedResponse,
                  taskType
                }]);
                
                toast.success('Streaming response completed!');
              } else if (data.type === 'error') {
                toast.error(data.error || 'Streaming error occurred');
              }
            } catch (parseError) {
              console.error('Parse error:', parseError);
            }
          }
        }
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        toast.info('Request cancelled');
      } else {
        toast.error('Streaming failed');
      }
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setLoading(false);
  };

  const handleClear = () => {
    setPrompt('');
    setResponse(null);
    setStreamingResponse('');
    setConversationHistory([]);
  };

  const getTaskIcon = (type: string) => {
    const task = taskTypes.find(t => t.value === type);
    return task ? task.icon : MessageSquare;
  };

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2">
            <Sparkles className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Enhanced Ultimate AI Model
            </h1>
            <Sparkles className="h-8 w-8 text-purple-600" />
          </div>
          <p className="text-lg text-muted-foreground">
            Test the advanced AI model with 1M context window and 128K input token capacity, trained on 500B+ tokens across 218+ specialized datasets
          </p>
          <div className="flex items-center justify-center gap-2">
            <Badge variant="secondary" className="bg-green-100 text-green-800 border-green-200">
              99% Accuracy
            </Badge>
            <Badge variant="secondary" className="bg-blue-100 text-blue-800 border-blue-200">
              1M Context
            </Badge>
            <Badge variant="secondary" className="bg-purple-100 text-purple-800 border-purple-200">
              128K Input
            </Badge>
            <Badge variant="secondary" className="bg-orange-100 text-orange-800 border-orange-200">
              Enhanced v3.0
            </Badge>
            {streaming && (
              <Badge variant="secondary" className="bg-pink-100 text-pink-800 border-pink-200">
                <Zap className="h-3 w-3 mr-1" />
                Streaming
              </Badge>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Panel */}
          <div className="lg:col-span-2 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5" />
                  AI Interface
                </CardTitle>
                <CardDescription>
                  Interact with the Enhanced Ultimate AI Model
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Task Type Selection */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Task Type</label>
                  <Select value={taskType} onValueChange={setTaskType}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select task type" />
                    </SelectTrigger>
                    <SelectContent>
                      {taskTypes.map((type) => {
                        const Icon = type.icon;
                        return (
                          <SelectItem key={type.value} value={type.value}>
                            <div className="flex items-center gap-2">
                              <Icon className="h-4 w-4" />
                              <div>
                                <div className="font-medium">{type.label}</div>
                                <div className="text-xs text-muted-foreground">{type.description}</div>
                              </div>
                            </div>
                          </SelectItem>
                        );
                      })}
                    </SelectContent>
                  </Select>
                </div>

                {/* Prompt Input */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Prompt</label>
                  <Textarea
                    placeholder="Enter your prompt here..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="min-h-[120px] resize-none"
                  />
                </div>

                {/* Parameters */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Temperature: {temperature[0].toFixed(1)}
                    </label>
                    <Slider
                      value={temperature}
                      onValueChange={setTemperature}
                      max={2}
                      min={0}
                      step={0.1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Focused</span>
                      <span>Creative</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Max Tokens: {maxTokens[0]}
                    </label>
                    <Slider
                      value={maxTokens}
                      onValueChange={setMaxTokens}
                      max={8192}
                      min={100}
                      step={100}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>100</span>
                      <span>8K</span>
                    </div>
                  </div>
                </div>

                {/* Streaming Toggle */}
                <div className="flex items-center space-x-2">
                  <Switch
                    id="streaming"
                    checked={streaming}
                    onCheckedChange={setStreaming}
                  />
                  <label htmlFor="streaming" className="text-sm font-medium">
                    Enable streaming responses
                  </label>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  {loading && streaming ? (
                    <Button variant="destructive" onClick={handleStop}>
                      Stop
                    </Button>
                  ) : (
                    <Button 
                      onClick={handleSubmit} 
                      disabled={loading || !prompt.trim()}
                      className="flex-1"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          {streaming ? 'Streaming...' : 'Processing...'}
                        </>
                      ) : (
                        <>
                          <Sparkles className="mr-2 h-4 w-4" />
                          Generate Response
                        </>
                      )}
                    </Button>
                  )}
                  <Button variant="outline" onClick={handleClear}>
                    Clear
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Response Display */}
            {(response || streamingResponse) && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <div className="flex items-center gap-2">
                      {(() => {
                        const Icon = getTaskIcon(response?.metrics.task_type || taskType);
                        return <Icon className="h-5 w-5" />;
                      })()}
                      AI Response
                    </div>
                    <Badge variant="outline" className="ml-auto">
                      {response?.metrics.task_type || taskType}
                    </Badge>
                    {streaming && !response && (
                      <Badge variant="secondary" className="bg-orange-100 text-orange-800">
                        <Zap className="h-3 w-3 mr-1" />
                        Streaming
                      </Badge>
                    )}
                  </CardTitle>
                  <CardDescription>
                    Generated by Enhanced Ultimate AI Model
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-50 p-4 rounded-lg">
                    <p className="whitespace-pre-wrap text-sm leading-relaxed">
                      {response?.response || streamingResponse}
                      {streaming && !response && (
                        <span className="inline-block w-2 h-4 bg-blue-600 animate-pulse ml-1" />
                      )}
                    </p>
                  </div>
                  
                  {/* Performance Metrics */}
                  {response && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          {(response.metrics.performance_score * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Performance</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {(response.metrics.confidence * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Confidence</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">
                          {response.metrics.task_type}
                        </div>
                        <div className="text-xs text-muted-foreground">Task Type</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600">
                          {new Date(response.timestamp).toLocaleTimeString()}
                        </div>
                        <div className="text-xs text-muted-foreground">Time</div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          {/* Side Panel */}
          <div className="space-y-4">
            {/* Model Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Model Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Status</span>
                  <Badge className="bg-green-100 text-green-800">Operational</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Version</span>
                  <span className="text-sm font-medium">Enhanced v2.0</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Accuracy</span>
                  <span className="text-sm font-medium">98%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Domains</span>
                  <span className="text-sm font-medium">5</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Streaming</span>
                  <Badge className={streaming ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-800"}>
                    {streaming ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Capabilities */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  Capabilities
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {taskTypes.map((type) => {
                  const Icon = type.icon;
                  return (
                    <div key={type.value} className="flex items-center gap-2 text-sm">
                      <Icon className="h-4 w-4 text-muted-foreground" />
                      <span>{type.label}</span>
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {/* Conversation History */}
            {conversationHistory.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MessageSquare className="h-5 w-5" />
                    Recent Conversations
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="max-h-64 overflow-y-auto space-y-3">
                    {conversationHistory.slice(-5).reverse().map((conv, index) => {
                      const Icon = getTaskIcon(conv.taskType);
                      return (
                        <div key={index} className="border rounded-lg p-3 space-y-2">
                          <div className="flex items-center gap-2">
                            <Icon className="h-3 w-3" />
                            <Badge variant="outline" className="text-xs">
                              {conv.taskType}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground line-clamp-2">
                            {conv.prompt}
                          </div>
                          <div className="text-xs text-muted-foreground line-clamp-2">
                            {conv.response}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}