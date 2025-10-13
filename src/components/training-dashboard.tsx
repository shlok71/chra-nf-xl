'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { 
  Play, 
  Pause, 
  Square, 
  RefreshCw, 
  TrendingUp, 
  Clock, 
  Database, 
  Brain,
  CheckCircle,
  XCircle,
  AlertCircle,
  BarChart3,
  Settings,
  Zap,
  Target,
  Award
} from 'lucide-react';
import { toast } from 'sonner';

interface TrainingJob {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: {
    currentEpoch: number;
    currentStep: number;
    totalSteps: number;
    loss: number;
    perplexity: number;
    learningRate: number;
    datasetProgress: number;
    timeElapsed: number;
    estimatedTimeRemaining: number;
  };
  datasets: Array<{
    name: string;
    samples: number;
    tokens: number;
    type: string;
  }>;
  startTime: Date;
  endTime?: Date;
  results?: {
    finalLoss: number;
    finalPerplexity: number;
    trainingTime: number;
    modelImprovements: {
      accuracyGain: number;
      coherenceGain: number;
      knowledgeGain: number;
      reasoningGain: number;
    };
  };
  error?: string;
}

export default function TrainingDashboard() {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [statistics, setStatistics] = useState<any>(null);
  const [newJobForm, setNewJobForm] = useState({
    name: '',
    description: '',
    preset: 'production',
    datasetTypes: ['text', 'code', 'math', 'reasoning']
  });

  useEffect(() => {
    fetchTrainingJobs();
    const interval = setInterval(fetchTrainingJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchTrainingJobs = async () => {
    try {
      const response = await fetch('/api/ai/training');
      const data = await response.json();
      if (data.success) {
        setJobs(data.jobs);
        setStatistics(data.statistics);
      }
    } catch (error) {
      console.error('Failed to fetch training jobs:', error);
    }
  };

  const createTrainingJob = async () => {
    if (!newJobForm.name || !newJobForm.description) {
      toast.error('Please provide a name and description for the training job');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/ai/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newJobForm)
      });

      const data = await response.json();
      if (data.success) {
        toast.success('Training job created successfully');
        setNewJobForm({ name: '', description: '', preset: 'production', datasetTypes: ['text', 'code', 'math', 'reasoning'] });
        fetchTrainingJobs();
      } else {
        toast.error(data.error || 'Failed to create training job');
      }
    } catch (error) {
      toast.error('Failed to create training job');
    } finally {
      setLoading(false);
    }
  };

  const startTrainingJob = async (jobId: string) => {
    try {
      const response = await fetch('/api/ai/training', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId, action: 'start' })
      });

      const data = await response.json();
      if (data.success) {
        toast.success('Training job started');
        fetchTrainingJobs();
      } else {
        toast.error(data.error || 'Failed to start training job');
      }
    } catch (error) {
      toast.error('Failed to start training job');
    }
  };

  const stopTrainingJob = async (jobId: string) => {
    try {
      const response = await fetch('/api/ai/training', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId, action: 'stop' })
      });

      const data = await response.json();
      if (data.success) {
        toast.success('Training job stopped');
        fetchTrainingJobs();
      } else {
        toast.error(data.error || 'Failed to stop training job');
      }
    } catch (error) {
      toast.error('Failed to stop training job');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'cancelled':
        return <Square className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  const formatTime = (milliseconds: number) => {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000000) return `${(num / 1000000000).toFixed(1)}B`;
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <div className="space-y-6">
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
          AI Model Training Dashboard
        </h1>
        <p className="text-gray-600">
          Train and fine-tune AI models on massive datasets
        </p>
      </div>

      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Jobs</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.totalJobs}</div>
              <p className="text-xs text-muted-foreground">
                {statistics.activeJobs} active, {statistics.completedJobs} completed
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Training Samples</CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(statistics.totalTrainingSamples)}</div>
              <p className="text-xs text-muted-foreground">
                {formatNumber(statistics.totalTrainingTokens)} tokens
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {statistics.totalJobs > 0 ? Math.round((statistics.successfulJobs / statistics.totalJobs) * 100) : 0}%
              </div>
              <p className="text-xs text-muted-foreground">
                {statistics.successfulJobs} successful, {statistics.failedJobs} failed
              </p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Training Time</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatTime(statistics.averageTrainingTime)}</div>
              <p className="text-xs text-muted-foreground">
                Per completed job
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="jobs" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="jobs">Training Jobs</TabsTrigger>
          <TabsTrigger value="create">Create Job</TabsTrigger>
        </TabsList>

        <TabsContent value="jobs" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {jobs.map((job) => (
              <Card key={job.id} className="cursor-pointer hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(job.status)}
                      <CardTitle className="text-lg">{job.name}</CardTitle>
                    </div>
                    <Badge className={getStatusColor(job.status)}>
                      {job.status}
                    </Badge>
                  </div>
                  <CardDescription>{job.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {job.status === 'running' && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{Math.round((job.progress.currentStep / job.progress.totalSteps) * 100)}%</span>
                        </div>
                        <Progress value={(job.progress.currentStep / job.progress.totalSteps) * 100} />
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Step {job.progress.currentStep}/{job.progress.totalSteps}</span>
                          <span>Epoch {job.progress.currentEpoch}</span>
                        </div>
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Loss:</span>
                        <span className="ml-2">{job.progress.loss.toFixed(4)}</span>
                      </div>
                      <div>
                        <span className="font-medium">Perplexity:</span>
                        <span className="ml-2">{job.progress.perplexity.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="font-medium">Datasets:</span>
                        <span className="ml-2">{job.datasets.length}</span>
                      </div>
                      <div>
                        <span className="font-medium">Samples:</span>
                        <span className="ml-2">{formatNumber(job.datasets.reduce((sum, ds) => sum + ds.samples, 0))}</span>
                      </div>
                    </div>

                    {job.results && (
                      <div className="space-y-2">
                        <h4 className="font-medium text-sm">Model Improvements</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span>Accuracy:</span>
                            <span>+{(job.results.modelImprovements.accuracyGain * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Coherence:</span>
                            <span>+{(job.results.modelImprovements.coherenceGain * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Knowledge:</span>
                            <span>+{(job.results.modelImprovements.knowledgeGain * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Reasoning:</span>
                            <span>+{(job.results.modelImprovements.reasoningGain * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="flex gap-2">
                      {job.status === 'pending' && (
                        <Button
                          onClick={() => startTrainingJob(job.id)}
                          size="sm"
                          className="flex-1"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Start
                        </Button>
                      )}
                      {job.status === 'running' && (
                        <Button
                          onClick={() => stopTrainingJob(job.id)}
                          size="sm"
                          variant="destructive"
                          className="flex-1"
                        >
                          <Square className="h-4 w-4 mr-2" />
                          Stop
                        </Button>
                      )}
                      <Button
                        onClick={() => setSelectedJob(job)}
                        size="sm"
                        variant="outline"
                        className="flex-1"
                      >
                        <Settings className="h-4 w-4 mr-2" />
                        Details
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="create" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Create New Training Job</CardTitle>
              <CardDescription>
                Configure and start a new AI model training job
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Job Name</label>
                  <Input
                    value={newJobForm.name}
                    onChange={(e) => setNewJobForm({ ...newJobForm, name: e.target.value })}
                    placeholder="Enter job name"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Preset</label>
                  <Select
                    value={newJobForm.preset}
                    onValueChange={(value) => setNewJobForm({ ...newJobForm, preset: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="quickTest">Quick Test</SelectItem>
                      <SelectItem value="production">Production</SelectItem>
                      <SelectItem value="research">Research</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Description</label>
                <Textarea
                  value={newJobForm.description}
                  onChange={(e) => setNewJobForm({ ...newJobForm, description: e.target.value })}
                  placeholder="Describe the training job objective"
                  rows={3}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Dataset Types</label>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {['text', 'code', 'math', 'reasoning', 'conversation', 'knowledge', 'multimodal'].map((type) => (
                    <label key={type} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={newJobForm.datasetTypes.includes(type)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setNewJobForm({
                              ...newJobForm,
                              datasetTypes: [...newJobForm.datasetTypes, type]
                            });
                          } else {
                            setNewJobForm({
                              ...newJobForm,
                              datasetTypes: newJobForm.datasetTypes.filter(t => t !== type)
                            });
                          }
                        }}
                        className="rounded"
                      />
                      <span className="text-sm capitalize">{type}</span>
                    </label>
                  ))}
                </div>
              </div>

              <Button
                onClick={createTrainingJob}
                disabled={loading || !newJobForm.name || !newJobForm.description}
                className="w-full"
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Create Training Job
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {selectedJob && (
        <Card className="fixed inset-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-0 overflow-auto">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-xl">{selectedJob.name}</CardTitle>
              <Button
                onClick={() => setSelectedJob(null)}
                variant="ghost"
                size="sm"
              >
                <XCircle className="h-4 w-4" />
              </Button>
            </div>
            <CardDescription>{selectedJob.description}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center gap-4">
              {getStatusIcon(selectedJob.status)}
              <Badge className={getStatusColor(selectedJob.status)}>
                {selectedJob.status}
              </Badge>
              <div className="text-sm text-muted-foreground">
                Started: {new Date(selectedJob.startTime).toLocaleString()}
              </div>
              {selectedJob.endTime && (
                <div className="text-sm text-muted-foreground">
                  Ended: {new Date(selectedJob.endTime).toLocaleString()}
                </div>
              )}
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Training Progress</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <span className="text-sm font-medium">Current Step</span>
                  <div className="text-2xl font-bold">{selectedJob.progress.currentStep}</div>
                  <div className="text-sm text-muted-foreground">of {selectedJob.progress.totalSteps}</div>
                </div>
                <div>
                  <span className="text-sm font-medium">Current Epoch</span>
                  <div className="text-2xl font-bold">{selectedJob.progress.currentEpoch}</div>
                </div>
                <div>
                  <span className="text-sm font-medium">Loss</span>
                  <div className="text-2xl font-bold">{selectedJob.progress.loss.toFixed(4)}</div>
                </div>
                <div>
                  <span className="text-sm font-medium">Perplexity</span>
                  <div className="text-2xl font-bold">{selectedJob.progress.perplexity.toFixed(2)}</div>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Training Datasets</h3>
              <div className="space-y-2">
                {selectedJob.datasets.map((dataset, index) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-muted rounded">
                    <div>
                      <div className="font-medium">{dataset.name}</div>
                      <div className="text-sm text-muted-foreground capitalize">{dataset.type}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">{formatNumber(dataset.samples)} samples</div>
                      <div className="text-sm text-muted-foreground">{formatNumber(dataset.tokens)} tokens</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {selectedJob.results && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Training Results</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <span className="text-sm font-medium">Final Loss</span>
                    <div className="text-2xl font-bold">{selectedJob.results.finalLoss.toFixed(4)}</div>
                  </div>
                  <div>
                    <span className="text-sm font-medium">Final Perplexity</span>
                    <div className="text-2xl font-bold">{selectedJob.results.finalPerplexity.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-sm font-medium">Training Time</span>
                    <div className="text-2xl font-bold">{formatTime(selectedJob.results.trainingTime)}</div>
                  </div>
                  <div>
                    <span className="text-sm font-medium">Epochs Completed</span>
                    <div className="text-2xl font-bold">{selectedJob.results.epochsCompleted}</div>
                  </div>
                </div>
              </div>
            )}

            {selectedJob.error && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-red-600">Error Information</h3>
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800">{selectedJob.error}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}