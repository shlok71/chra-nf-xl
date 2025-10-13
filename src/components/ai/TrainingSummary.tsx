'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  CheckCircle2, 
  TrendingUp, 
  Award, 
  Brain,
  Heart,
  Languages,
  Target,
  Sparkles,
  Download,
  Share
} from 'lucide-react';

interface TrainingSummaryProps {
  trainingResults: {
    trainingType: string;
    overallImprovement: string;
    improvements: any;
    modelVersion: string;
    timestamp: string;
  };
  onExport?: () => void;
  onShare?: () => void;
}

export default function TrainingSummary({ trainingResults, onExport, onShare }: TrainingSummaryProps) {
  const isUserPreference = trainingResults.trainingType === 'user-preference';
  
  const getIcon = () => {
    if (isUserPreference) {
      return <Heart className="h-8 w-8 text-pink-600" />;
    }
    return <Languages className="h-8 w-8 text-blue-600" />;
  };

  const getTitle = () => {
    if (isUserPreference) {
      return "User Preference Training Complete";
    }
    return "Multilingual Training Complete";
  };

  const getDescription = () => {
    if (isUserPreference) {
      return "AI model has been successfully trained on user response preferences and communication patterns";
    }
    return "AI model has been successfully trained on multilingual datasets and cultural contexts";
  };

  const metrics = isUserPreference ? [
    { label: "Response Satisfaction", value: trainingResults.improvements.responseSatisfaction, icon: Heart, color: "pink" },
    { label: "Conversation Flow", value: trainingResults.improvements.conversationFlow, icon: Target, color: "blue" },
    { label: "Personalization", value: trainingResults.improvements.personalization, icon: Brain, color: "green" },
    { label: "Clarity Score", value: trainingResults.improvements.clarityScore, icon: Sparkles, color: "yellow" },
    { label: "Helpfulness", value: trainingResults.improvements.helpfulness, icon: Award, color: "purple" },
    { label: "Adaptability", value: trainingResults.improvements.adaptability, icon: TrendingUp, color: "indigo" }
  ] : [
    { label: "Language Accuracy", value: trainingResults.improvements.languageAccuracy, icon: Languages, color: "blue" },
    { label: "Cultural Understanding", value: trainingResults.improvements.culturalUnderstanding, icon: Brain, color: "purple" },
    { label: "Dialect Processing", value: trainingResults.improvements.dialectProcessing, icon: Target, color: "green" },
    { label: "Cross-Lingual Transfer", value: trainingResults.improvements.crossLingualTransfer, icon: TrendingUp, color: "yellow" },
    { label: "Regional Adaptation", value: trainingResults.improvements.regionalAdaptation, icon: Heart, color: "indigo" },
    { label: "Technical Multilingual", value: trainingResults.improvements.technicalMultilingual, icon: Sparkles, color: "pink" }
  ];

  return (
    <div className="space-y-6">
      {/* Success Header */}
      <Card className="border-2 border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            {getIcon()}
          </div>
          <CardTitle className="text-2xl text-green-800 flex items-center justify-center gap-2">
            <CheckCircle2 className="h-6 w-6 text-green-600" />
            {getTitle()}
          </CardTitle>
          <CardDescription className="text-green-700 max-w-2xl mx-auto">
            {getDescription()}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex justify-center gap-4 flex-wrap">
            <Badge variant="secondary" className="bg-green-100 text-green-800 border-green-200">
              Model: {trainingResults.modelVersion}
            </Badge>
            <Badge variant="secondary" className="bg-blue-100 text-blue-800 border-blue-200">
              Overall Improvement: {trainingResults.overallImprovement}%
            </Badge>
            <Badge variant="secondary" className="bg-purple-100 text-purple-800 border-purple-200">
              {new Date(trainingResults.timestamp).toLocaleDateString()}
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-blue-600" />
            Performance Metrics
          </CardTitle>
          <CardDescription>
            Detailed improvements across all key performance indicators
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {metrics.map((metric, index) => {
              const Icon = metric.icon;
              const colorClasses = {
                pink: "from-pink-50 to-pink-100 border-pink-200 text-pink-800",
                blue: "from-blue-50 to-blue-100 border-blue-200 text-blue-800",
                green: "from-green-50 to-green-100 border-green-200 text-green-800",
                yellow: "from-yellow-50 to-yellow-100 border-yellow-200 text-yellow-800",
                purple: "from-purple-50 to-purple-100 border-purple-200 text-purple-800",
                indigo: "from-indigo-50 to-indigo-100 border-indigo-200 text-indigo-800"
              };
              
              return (
                <div 
                  key={index}
                  className={`p-4 rounded-lg border bg-gradient-to-br ${colorClasses[metric.color]}`}
                >
                  <div className="flex items-center gap-3 mb-2">
                    <Icon className="h-5 w-5" />
                    <span className="font-medium text-sm">{metric.label}</span>
                  </div>
                  <div className="text-2xl font-bold">
                    {metric.value}
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Training Impact */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Award className="h-5 w-5 text-purple-600" />
            Training Impact
          </CardTitle>
          <CardDescription>
            How this training enhances the AI model capabilities
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800">Immediate Benefits</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                {isUserPreference ? (
                  <>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Enhanced user satisfaction through personalized responses
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Improved conversation flow and engagement
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Better adaptation to user communication styles
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Increased clarity and helpfulness in responses
                    </li>
                  </>
                ) : (
                  <>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Improved accuracy across multiple languages
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Enhanced cultural context understanding
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Better handling of regional dialects and variations
                    </li>
                    <li className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      Stronger cross-lingual knowledge transfer
                    </li>
                  </>
                )}
              </ul>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold text-gray-800">Long-term Advantages</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                {isUserPreference ? (
                  <>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Continuous improvement from user interactions
                    </li>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Scalable personalization across user base
                    </li>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Reduced need for manual response adjustments
                    </li>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Higher user retention and satisfaction rates
                    </li>
                  </>
                ) : (
                  <>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Expanded global reach and accessibility
                    </li>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Reduced language barriers in communication
                    </li>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Better support for diverse user communities
                    </li>
                    <li className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      Enhanced cross-cultural collaboration capabilities
                    </li>
                  </>
                )}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex justify-center gap-4">
        <Button 
          onClick={onExport}
          variant="outline"
          className="flex items-center gap-2"
        >
          <Download className="h-4 w-4" />
          Export Results
        </Button>
        <Button 
          onClick={onShare}
          className="flex items-center gap-2"
        >
          <Share className="h-4 w-4" />
          Share Training Report
        </Button>
      </div>
    </div>
  );
}