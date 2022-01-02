package com.example.uos.project;
import java.io.IOException;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.media.Image;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceHolder;
import android.hardware.Camera;
import android.hardware.Camera.Parameters;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageView;

class CameraSurface implements SurfaceHolder.Callback, TextureView.SurfaceTextureListener
{
    private Camera mCamera = null;
    private TextureView mtextureview = null;

    private int width;
    private int height;
    private View imgView;

    private boolean bProcessing = false;

    CameraSurface(int width, int height, View imgView, Camera camera, TextureView textureView)
    {
        this.width = width;
        this.height = height;
        this.imgView = imgView;
        this.mCamera = camera;
        this.mtextureview = textureView;
    }

    @Override
    public void surfaceChanged(SurfaceHolder arg0, int arg1, int arg2, int arg3)
    {
        //Parameters parameters = mCamera.getParameters();
        //parameters.setPreviewSize(width, height);

        //mCamera.setParameters(parameters);
        //mCamera.startPreview();
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder)
    {
        /*mCamera.setDisplayOrientation(180);
        try
        {
            mCamera.setPreviewDisplay(surfaceHolder);
        }
        catch (IOException e)
        {
            mCamera.release();
            mCamera = null;
        }*/
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder arg0)
    {
        /*mCamera.setPreviewCallback(null);
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;*/
    }

    public void stopPreview(){
        mCamera.stopPreview();
    }
    public void resumePreview(){
        mCamera.startPreview();
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i, int i1) {
        Log.d("POE","in");
        Matrix matrix = new Matrix();
        matrix.setScale(1,-1);
        matrix.postTranslate(0, height);
        mtextureview.setTransform(matrix);

        Parameters parameters = mCamera.getParameters();
        parameters.setPreviewSize(width, height);
        mCamera.setDisplayOrientation(180);
        mCamera.setParameters(parameters);
        try {
            mCamera.setPreviewTexture(surfaceTexture);
            mCamera.startPreview();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i, int i1) {

    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
        mCamera.setPreviewCallback(null);
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;
        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {

    }
}