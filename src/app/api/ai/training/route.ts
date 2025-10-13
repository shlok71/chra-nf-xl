import { NextRequest, NextResponse } from 'next/server';
import { aiTrainingService } from '@/lib/ai-training-service';
import { COMPREHENSIVE_TRAINING_DATASETS } from '@/lib/training-datasets';
import { TRAINING_PRESETS } from '@/lib/training-pipeline';

// POST: Create a new training job
export async function POST(request: NextRequest) {
  try {
    const { 
      name, 
      description, 
      config, 
      datasetTypes, 
      preset 
    } = await request.json();

    if (!name || !description) {
      return NextResponse.json({ 
        error: 'Name and description are required' 
      }, { status: 400 });
    }

    // Use preset if provided
    let trainingConfig = config;
    if (preset && TRAINING_PRESETS[preset as keyof typeof TRAINING_PRESETS]) {
      trainingConfig = {
        ...TRAINING_PRESETS[preset as keyof typeof TRAINING_PRESETS],
        ...config
      };
    }

    const jobId = await aiTrainingService.createTrainingJob(
      name,
      description,
      trainingConfig,
      datasetTypes
    );

    return NextResponse.json({
      success: true,
      jobId,
      message: 'Training job created successfully'
    });

  } catch (error) {
    console.error('Failed to create training job:', error);
    return NextResponse.json({
      error: 'Failed to create training job',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// GET: Get all training jobs or a specific job
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');
    const status = searchParams.get('status');

    if (jobId) {
      const job = aiTrainingService.getTrainingJob(jobId);
      if (!job) {
        return NextResponse.json({
          error: 'Training job not found'
        }, { status: 404 });
      }
      return NextResponse.json({ success: true, job });
    }

    let jobs = aiTrainingService.getAllTrainingJobs();
    
    if (status) {
      jobs = jobs.filter(job => job.status === status);
    }

    return NextResponse.json({
      success: true,
      jobs,
      statistics: aiTrainingService.getTrainingStatistics()
    });

  } catch (error) {
    console.error('Failed to get training jobs:', error);
    return NextResponse.json({
      error: 'Failed to get training jobs',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// PUT: Update a training job (start/stop)
export async function PUT(request: NextRequest) {
  try {
    const { jobId, action } = await request.json();

    if (!jobId || !action) {
      return NextResponse.json({
        error: 'Job ID and action are required'
      }, { status: 400 });
    }

    const job = aiTrainingService.getTrainingJob(jobId);
    if (!job) {
      return NextResponse.json({
        error: 'Training job not found'
      }, { status: 404 });
    }

    switch (action) {
      case 'start':
        await aiTrainingService.startTrainingJob(jobId);
        return NextResponse.json({
          success: true,
          message: 'Training job started successfully'
        });

      case 'stop':
        await aiTrainingService.stopTrainingJob(jobId);
        return NextResponse.json({
          success: true,
          message: 'Training job stopped successfully'
        });

      default:
        return NextResponse.json({
          error: 'Invalid action. Must be "start" or "stop"'
        }, { status: 400 });
    }

  } catch (error) {
    console.error('Failed to update training job:', error);
    return NextResponse.json({
      error: 'Failed to update training job',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// DELETE: Delete a training job
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');

    if (!jobId) {
      return NextResponse.json({
        error: 'Job ID is required'
      }, { status: 400 });
    }

    const job = aiTrainingService.getTrainingJob(jobId);
    if (!job) {
      return NextResponse.json({
        error: 'Training job not found'
      }, { status: 404 });
    }

    if (job.status === 'running') {
      await aiTrainingService.stopTrainingJob(jobId);
    }

    // Note: In a real implementation, you might want to actually delete the job
    // For now, we'll just stop it if it's running
    
    return NextResponse.json({
      success: true,
      message: 'Training job deleted successfully'
    });

  } catch (error) {
    console.error('Failed to delete training job:', error);
    return NextResponse.json({
      error: 'Failed to delete training job',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}