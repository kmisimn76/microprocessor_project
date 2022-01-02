package com.example.uos.project;

import android.graphics.Bitmap;
import android.util.Log;

public class JNIProcess implements JNIListener{
    static {
        System.loadLibrary("native-lib");
    }

    public static native void init(int width, int height);
    public static native int[] process(int width, int height,
                                       Bitmap bit, byte [] pixels, byte [] bold_pixels, int [] t_nums, int nums_n, int [] nums_group, int nums_group_n,
                                       int [] t_devs, int devs_n, int Threshold);
    public static native void matchinglabel(byte [] pixels, int width, int height, int [] t_nums, int [] nums_label, int nums_n,
                                       int [] t_devs, int [] devs_label, int [] devs_bipolar, int devs_n);

    public native static int openDriverSw(String path);
    public native static void closeDriverSw();
    public native static char readDriver();
    public native static int getInterrupt();

    public native static int openDriverSe(String path);
    public native static void closeDriverSe();
    public native static void writeDriver(byte[] data, int length);

    private boolean mConnectFlag;
    private TranseThread mTranseThread;
    private JNIListener mMainActivity;

    public JNIProcess() { mConnectFlag = false; }
    public void setListener(JNIListener m) { mMainActivity = m; }

    public int open(String driver){
        if(mConnectFlag == true) return -1;
        if(openDriverSw(driver)>0){
            mConnectFlag = true;
            mTranseThread = new TranseThread();
            mTranseThread.start();
            return 1;
        }
        else return -1;
    }
    public void close(){
        if(mTranseThread != null){
            mTranseThread = null;
        }
        if(mConnectFlag==false) return;
        mConnectFlag = false;
        closeDriverSw();
    }
    protected void finalize() throws Throwable {
        close();
        super.finalize();
    }
    public char read() { return readDriver(); }

    private class TranseThread extends Thread {
        public void run(){
            super.run();
            try{
                while(true){
                    try{
                        Log.e("test","waiting");
                        onReceive(getInterrupt());
                        Thread.sleep(100);
                    }catch (InterruptedException e){
                        e.printStackTrace();
                        break;
                    }
                }
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onReceive(int val) {
        if(mMainActivity != null){
            Log.e("test", "received");
            mMainActivity.onReceive(val);
        }
    }
}
