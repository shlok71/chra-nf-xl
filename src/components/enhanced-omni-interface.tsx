'use client';

import { useState, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Loader2, 
  Upload, 
  Image, 
  FileText, 
  Camera, 
  Download, 
  Eye, 
  Brain, 
  Palette, 
  Zap, 
  Settings, 
  X, 
  Check,
  TrendingUp,
  BarChart3,
  Calculator,
  DollarSign,
  Lightbulb,
  Sparkles,
  Rocket,
  Target,
  Award,
  Star
} from 'lucide-react';
import { toast } from 'sonner';

interface OmniResponse {
  success: boolean;
  response?: string;
  vision_analysis?: any;
  images?: any[];
  metadata: any;
  task?: string;
}

export default function EnhancedOmniInterface() {
  const [activeTab, setActiveTab] = useState('chat');
  const [prompt, setPrompt] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [analysisType, setAnalysisType] = useState('comprehensive');
  const [detail, setDetail] = useState('high');
  const [temperature, setTemperature] = useState([0.7]);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<OmniResponse | null>(null);
  
  // Vision Analysis States
  const [visionLoading, setVisionLoading] = useState(false);
  const [visionResponse, setVisionResponse] = useState<any>(null);
  
  // Image Generation States
  const [genPrompt, setGenPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [imageSize, setImageSize] = useState('1024x1024');
  const [quality, setQuality] = useState('hd');
  const [imageStyle, setImageStyle] = useState('natural');
  const [numImages, setNumImages] = useState(1);
  const [guidanceScale, setGuidanceScale] = useState([7.5]);
  const [genLoading, setGenLoading] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<any[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.size > 50 * 1024 * 1024) {
        toast.error('Image size must be less than 50MB');
        return;
      }

      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const handleImageRemove = useCallback(() => {
    setSelectedImage(null);
    setImagePreview('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  const handleVisionAnalysis = async () => {
    if (!selectedImage) {
      toast.error('Please select an image for vision analysis');
      return;
    }

    setVisionLoading(true);
    setVisionResponse(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);
      formData.append('analysisType', analysisType);
      formData.append('detail', detail);
      if (prompt) formData.append('prompt', prompt);

      const response = await fetch('/api/ai/vision', {
        method: 'POST',
        body: formData
      });

      const data: OmniResponse = await response.json();

      if (data.success) {
        setVisionResponse(data.vision_analysis);
        toast.success('Vision analysis completed!');
      } else {
        toast.error(data.error || 'Vision analysis failed');
      }
    } catch (error: any) {
      console.error('Error:', error);
      toast.error('Failed to analyze image');
    } finally {
      setVisionLoading(false);
    }
  };

  const handleImageGeneration = async () => {
    if (!genPrompt.trim()) {
      toast.error('Please enter a prompt for image generation');
      return;
    }

    setGenLoading(true);
    setGeneratedImages([]);

    try {
      const response = await fetch('/api/ai/image-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: genPrompt,
          negative_prompt: negativePrompt,
          size: imageSize,
          quality,
          style: imageStyle,
          count: numImages,
          guidance_scale: guidanceScale[0]
        })
      });

      const data = await response.json();

      if (data.success) {
        setGeneratedImages(data.images);
        toast.success(`Generated ${data.images.length} image(s) successfully!`);
      } else {
        toast.error(data.error || 'Image generation failed');
      }
    } catch (error: any) {
      console.error('Error:', error);
      toast.error('Failed to generate images');
    } finally {
      setGenLoading(false);
    }
  };

  const downloadImage = (imageData: any, index: number) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${imageData.base64}`;
    link.download = `generated_image_${index + 1}.png`;
    link.click();
  };

  const analysisTypes = [
    { value: 'comprehensive', label: 'Comprehensive', icon: Eye, description: 'Complete visual analysis' },
    { value: 'technical', label: 'Technical', icon: Settings, description: 'Technical examination' },
    { value: 'creative', label: 'Creative', icon: Palette, description: 'Artistic interpretation' },
    { value: 'scientific', label: 'Scientific', icon: Brain, description: 'Scientific analysis' },
    { value: 'financial', label: 'Financial', icon: TrendingUp, description: 'Financial data analysis' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
          Advanced AI Omni Model
        </h2>
        <p className="text-gray-600">
          Vision • Text • Image Generation • Financial Analysis
        </p>
        
        <div className="flex items-center justify-center gap-2 flex-wrap">
          <Badge variant="secondary" className="bg-purple-100 text-purple-800">
            <Eye className="h-3 w-3 mr-1" />
            Vision Analysis
          </Badge>
          <Badge variant="secondary" className="bg-green-100 text-green-800">
            <Palette className="h-3 w-3 mr-1" />
            Image Generation
          </Badge>
          <Badge variant="secondary" className="bg-orange-100 text-orange-800">
            <TrendingUp className="h-3 w-3 mr-1" />
            Financial Analysis
          </Badge>
          <Badge variant="secondary" className="bg-pink-100 text-pink-800">
            <Calculator className="h-3 w-3 mr-1" />
            Enhanced Math
          </Badge>
          <Badge variant="secondary" className="bg-indigo-100 text-indigo-800">
            <Brain className="h-3 w-3 mr-1" />
            Superior Reasoning
          </Badge>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="vision" className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Vision
          </TabsTrigger>
          <TabsTrigger value="generate" className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Generate
          </TabsTrigger>
          <TabsTrigger value="analysis" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Analysis
          </TabsTrigger>
        </TabsList>

        {/* Vision Analysis Tab */}
        <TabsContent value="vision" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Advanced Vision Analysis
              </CardTitle>
              <CardDescription>
                State-of-the-art image understanding with multiple analysis types
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Image Upload */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Upload Image</label>
                <div className="flex items-center gap-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                  />
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    variant="outline"
                    disabled={visionLoading}
                  >
                    <Camera className="h-4 w-4 mr-2" />
                    Choose Image
                  </Button>
                  {selectedImage && (
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">
                        <Image className="h-3 w-3 mr-1" />
                        {selectedImage.name}
                      </Badge>
                      <Button
                        onClick={handleImageRemove}
                        variant="ghost"
                        size="sm"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>
                
                {imagePreview && (
                  <div className="mt-2">
                    <img
                      src={imagePreview}
                      alt="Uploaded image preview"
                      className="max-w-full h-64 object-cover rounded-lg border"
                    />
                  </div>
                )}
              </div>

              {/* Analysis Type */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Analysis Type</label>
                <Select value={analysisType} onValueChange={setAnalysisType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {analysisTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        <div>
                          <div className="font-medium">{type.label}</div>
                          <div className="text-sm text-gray-600">{type.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Detail Level */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Detail Level</label>
                <Select value={detail} onValueChange={setDetail}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="standard">Standard</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="maximum">Maximum</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Prompt */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Analysis Prompt (Optional)</label>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Specific instructions for the analysis..."
                  rows={2}
                  disabled={visionLoading}
                />
              </div>

              {/* Analyze Button */}
              <Button
                onClick={handleVisionAnalysis}
                disabled={visionLoading || !selectedImage}
                className="w-full"
              >
                {visionLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4 mr-2" />
                    Analyze Image
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Vision Results */}
          {visionResponse && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Check className="h-5 w-5 text-green-600" />
                  Vision Analysis Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose max-w-none">
                  <pre className="whitespace-pre-wrap bg-gray-50 p-4 rounded-lg text-sm">
                    {JSON.stringify(visionResponse, null, 2)}
                  </pre>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Image Generation Tab */}
        <TabsContent value="generate" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Palette className="h-5 w-5" />
                AI Image Generation
              </CardTitle>
              <CardDescription>
                Create stunning images with advanced AI models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Prompt */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Prompt</label>
                <Textarea
                  value={genPrompt}
                  onChange={(e) => setGenPrompt(e.target.value)}
                  placeholder="Describe the image you want to generate..."
                  rows={3}
                  disabled={genLoading}
                />
              </div>

              {/* Negative Prompt */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Negative Prompt (Optional)</label>
                <Textarea
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="What you don't want in the image..."
                  rows={2}
                  disabled={genLoading}
                />
              </div>

              {/* Settings Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Image Size */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Image Size</label>
                  <Select value={imageSize} onValueChange={setImageSize}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1024x1024">1024x1024 (Square)</SelectItem>
                      <SelectItem value="1792x1024">1792x1024 (Landscape)</SelectItem>
                      <SelectItem value="1024x1792">1024x1792 (Portrait)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Quality */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Quality</label>
                  <Select value={quality} onValueChange={setQuality}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="standard">Standard</SelectItem>
                      <SelectItem value="hd">HD</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Style */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Style</label>
                  <Select value={imageStyle} onValueChange={setImageStyle}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="natural">Natural</SelectItem>
                      <SelectItem value="vivid">Vivid</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Number of Images */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Number of Images</label>
                  <Select value={numImages.toString()} onValueChange={(v) => setNumImages(parseInt(v))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1 Image</SelectItem>
                      <SelectItem value="2">2 Images</SelectItem>
                      <SelectItem value="3">3 Images</SelectItem>
                      <SelectItem value="4">4 Images</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Guidance Scale */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Guidance Scale: {guidanceScale[0]}</label>
                <Slider
                  value={guidanceScale}
                  onValueChange={setGuidanceScale}
                  max={20}
                  min={1}
                  step={0.5}
                  className="w-full"
                />
              </div>

              {/* Generate Button */}
              <Button
                onClick={handleImageGeneration}
                disabled={genLoading || !genPrompt.trim()}
                className="w-full"
              >
                {genLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate Images
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Generated Images */}
          {generatedImages.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Check className="h-5 w-5 text-green-600" />
                  Generated Images
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {generatedImages.map((image, index) => (
                    <div key={index} className="space-y-2">
                      <img
                        src={`data:image/png;base64,${image.base64}`}
                        alt={`Generated image ${index + 1}`}
                        className="w-full h-64 object-cover rounded-lg border"
                      />
                      <Button
                        onClick={() => downloadImage(image, index)}
                        variant="outline"
                        size="sm"
                        className="w-full"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download Image {index + 1}
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Advanced AI Analysis
              </CardTitle>
              <CardDescription>
                Comprehensive analysis with superior reasoning capabilities
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Temperature */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Temperature: {temperature[0]}</label>
                <Slider
                  value={temperature}
                  onValueChange={setTemperature}
                  max={2}
                  min={0}
                  step={0.1}
                  className="w-full"
                />
                <p className="text-xs text-gray-600">
                  Lower values make responses more focused and deterministic
                </p>
              </div>

              {/* Analysis Prompt */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Analysis Request</label>
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your analysis request, question, or task..."
                  rows={6}
                  disabled={loading}
                />
              </div>

              {/* Analyze Button */}
              <Button
                onClick={async () => {
                  if (!prompt.trim()) {
                    toast.error('Please enter an analysis request');
                    return;
                  }

                  setLoading(true);
                  setResponse(null);

                  try {
                    const response = await fetch('/api/ai/analyze', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        prompt: prompt.trim(),
                        temperature: temperature[0],
                        analysis_type: analysisType
                      })
                    });

                    const data: OmniResponse = await response.json();

                    if (data.success) {
                      setResponse(data);
                      toast.success('Analysis completed successfully!');
                    } else {
                      toast.error(data.error || 'Analysis failed');
                    }
                  } catch (error: any) {
                    console.error('Error:', error);
                    toast.error('Failed to perform analysis');
                  } finally {
                    setLoading(false);
                  }
                }}
                disabled={loading || !prompt.trim()}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-2" />
                    Perform Analysis
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Analysis Results */}
          {response && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Check className="h-5 w-5 text-green-600" />
                  Analysis Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose max-w-none">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="whitespace-pre-wrap">{response.response}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}