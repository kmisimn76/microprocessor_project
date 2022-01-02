package com.example.uos.project;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Picture;
import android.hardware.Camera;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;

public class MainActivity extends AppCompatActivity implements JNIListener {

    private Camera mCamera;
    CameraSurface cameraSurface;
    int previewSizeWidth = 640;
    int previewSizeHeight = 480;
    DrawView imgView;
    boolean Pictured;
    TextView resultText;

    private int findFronSideCamera(){
        int cameraid = -1;
        int num = Camera.getNumberOfCameras();

        for(int i=0;i<num;i++){
            Camera.CameraInfo cm = new Camera.CameraInfo();
            Camera.getCameraInfo(i, cm);
            if(cm.facing==Camera.CameraInfo.CAMERA_FACING_FRONT){
                cameraid = i;
                break;
            }
        }
        return cameraid;
    }

    int Threshold = 26;
    int Voltage = 5;
    boolean state = true;
    public Handler handler = new Handler(){
        @Override
        public void handleMessage(Message msg){
            switch(msg.arg1){
                case 1:
                    //tv.setText("Up");
                    if(Threshold<99) Threshold+=3;
                    break;
                case 2:
                    //tv.setText("Donw");
                    if(Threshold>=10) Threshold-=3;
                    break;
                case 3:
                    //tv.setText("Left");
                    if(Voltage>=1) Voltage-=1;
                    break;
                case 4:
                    //tv.setText("Right");
                    if(Voltage<99) Voltage+=1;
                    break;
                case 5:
                    //tv.setText("Center");
                    if(state==true){
                        state = false;
                    }
                    else {
                        state = true;
                    }
                    break;
            }
        }
    };

    @Override
    public void onReceive(int val) {
        Message msg = Message.obtain();
        msg.arg1 = val;
        handler.sendMessage(msg);
    }

    @Override
    protected void onPause() {
        JNIProcess.closeDriverSe();
        mThreadRun=false;
        mStart = false;
        mSegThread=null;
        mDriver.close();
        super.onPause();
    }
    @Override
    protected void onResume() {
        if(JNIProcess.openDriverSe("/dev/sm9s5422_segmentp")<0){
            Toast.makeText(MainActivity.this, "segment open error",Toast.LENGTH_SHORT).show();
        }
        mThreadRun=true;
        mStart = true;
        mSegThread = new SegmentThread();
        mSegThread.start();
        if(mDriver.open("/dev/sm9s5422_interrupt")<0)
            Toast.makeText(MainActivity.this, "Interrupt Open Failed", Toast.LENGTH_SHORT).show();
        super.onResume();
    }

    boolean mThreadRun, mStart;
    SegmentThread mSegThread;
    private class SegmentThread extends Thread {
        @Override
        public void run() {
            super.run();
            while(mThreadRun) {
                byte[] n = {0, 0, 0, 0, 0, 0, 0};

                if(mStart==false) { JNIProcess.writeDriver(n, n.length); }
                else {
                    if(state==true) {
                        n[0] = (byte) 32;
                        n[1] = (byte) 33;
                        n[2] = (byte) 34;
                        n[3] = (byte) 0;
                        n[4] = (byte) (Voltage % 100 / 10);
                        n[5] = (byte) (Voltage % 10);
                    }
                    else {
                        n[0] = (byte) 35;
                        n[1] = (byte) 36;
                        n[2] = (byte) 37;
                        n[3] = (byte) 0;
                        n[4] = (byte) (Threshold % 100 / 10);
                        n[5] = (byte) (Threshold % 10);
                    }
                    JNIProcess.writeDriver(n, n.length);
                }
            }
        }
    }

    private class DrawView extends View {
        Bitmap bmp;
        Paint mPaint;

        public DrawView(Context context) {
            super(context);
            bmp = null;
            mPaint = new Paint();
        }
        public void SetBitmap(Bitmap bitmap){
            bmp = bitmap;
        }

        protected void onDraw(Canvas canvas){
            if(bmp!=null) {
                canvas.drawBitmap(bmp, 0, 0, null);
                mPaint.setColor(Color.GREEN);
                mPaint.setStyle(Paint.Style.STROKE);
                for (int i = 0; i < nums_n; i++) {
                    Log.d("POE", t_nums[i * 4]+","+t_nums[i * 4 + 3]);
                    canvas.drawRect(t_nums[i * 4], t_nums[i * 4 + 1], t_nums[i * 4 + 2], t_nums[i * 4 + 3], mPaint);
                    canvas.drawText(""+nums_label[i], t_nums[i * 4 + 2],t_nums[i * 4 + 3],mPaint);
                }
                mPaint.setColor(Color.RED);
                for (int i = 0; i < devs_n; i++) {
                    if(t_devs[i*4]>=0){
                        Log.d("POE", t_devs[i * 4]+","+t_devs[i * 4 + 2]+","+t_devs[i * 4 + 1]+","+t_devs[i * 4 + 3]);
                        canvas.drawRect(t_devs[i * 4], t_devs[i * 4 + 1], t_devs[i * 4 + 2], t_devs[i * 4 + 3], mPaint);
                        canvas.drawText(""+devs_label[i], t_devs[i * 4 + 2],t_devs[i * 4 + 3],mPaint);
                    }
                }
                //mPaint.setColor(Color.GREEN);
                //mPaint.setStyle(Paint.Style.STROKE);
                /*for(int i=0;i<devs_n;i++){
                    for(int j=i+1; j<devs_n; j++){
                        if(circuit_edge[i*devs_n+j]==1) {
                            int cx1, cx2, cy1, cy2;
                            cx1 = (t_devs[i * 4 + 2] + t_devs[i * 4]) / 2;
                            cx2 = (t_devs[i * 4 + 3] + t_devs[i * 4 + 1]) / 2;
                            cy1 = (t_devs[j * 4 + 2] + t_devs[j * 4]) / 2;
                            cy2 = (t_devs[j * 4 + 3] + t_devs[j * 4 + 1]) / 2;
                            canvas.drawLine(cx1,cx2,cy1,cy2,mPaint);
                        }
                    }
                }*/
                mPaint.setColor(Color.YELLOW);
                for(int i=0;i<devs_n;i++){
                    if(t_devs[i*4]>=0){
                        String type;
                        switch(circuit_devices[i][0]){
                            case 0:
                                type = "R";
                                break;
                            case 1:
                                type = "V";
                                break;
                            case 2:
                                type = "Ob";
                                break;
                            default:
                                type = "?";
                        }
                        canvas.drawText(type+":"+circuit_devices[i][1], t_devs[i * 4 + 2],t_devs[i * 4 + 3],mPaint);
                    }
                }

            }
        }
    }

    JNIProcess mDriver;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mDriver = new JNIProcess();
        mDriver.setListener(this);

        imgView = new DrawView(this);

        // Create camera preview
        TextureView mtextureview = new TextureView(this);
        //SurfaceView camView = new SurfaceView(this);
        //SurfaceHolder camHolder = camView.getHolder();
        //camView.setZOrderMediaOverlay(true);
        ;
        mCamera = Camera.open(findFronSideCamera());
        cameraSurface = new CameraSurface(previewSizeWidth, previewSizeHeight, imgView, mCamera,mtextureview);
        mtextureview.setSurfaceTextureListener(cameraSurface);
        //camHolder.addCallback(cameraSurface);
        //camHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        // Frame Layout for display
        FrameLayout mainLayout = findViewById(R.id.frame_layout);
        mainLayout.addView(mtextureview, new ViewGroup.LayoutParams(previewSizeWidth, previewSizeHeight));
        mainLayout.addView(imgView, new ViewGroup.LayoutParams(previewSizeWidth, previewSizeHeight));

        Pictured = false;
        Button btn1 = (Button) findViewById(R.id.button_capture);
        Log.e("mcamera","startprog");
        btn1.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view) {
                Log.e("mcamera","picted");
                Pictured = true;
                mCamera.takePicture(null, null, pictureCallback);
            }
        });
        Button btn2 = (Button) findViewById(R.id.button_restart);
        btn2.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view) {
                if(Pictured==true){
                    cameraSurface.resumePreview();
                    Pictured=false;
                    imgView.SetBitmap(null);
                    imgView.setVisibility(View.GONE);
                    Log.e("mcamera","resume");
                }
            }
        });

        resultText = (TextView) findViewById(R.id.textview_result);
    }


    private byte[] pixels = new byte[previewSizeWidth * previewSizeHeight * 4];
    private byte[] bold_pixels = new byte[previewSizeWidth * previewSizeHeight];
    private int[] t_nums = new int[200];
    private int[] nums_label = new int [200];
    private int   nums_n = 0;
    private int[] nums_group = new int [50];
    private int   nums_group_n = 0;
    private int[] t_devs = new int[200];
    private int[] devs_label = new int [200];
    private int[] devs_bipolar = new int[200];
    private int   devs_n = 0;

    private int[][] number_group = new int[50][4]; //***: num, center x, center y, n
    private int[][] circuit_devices = new int[50][4]; //***: type, number, node
    //private int[] circuit_edge = new int[50*50]; //***: edge table
    private int[][] analysis_node = new int[100][50]; ///device 배열 개수 바꾸기
    private int analysis_node_n = 0;

    private double[] circuit_constant = new double[50]; //회로값
    private double[] analysis_result = new double[50];

    double dist(int a, int b, int c, int d){
        return Math.sqrt(Math.pow((double)(a-c),2)+Math.pow((double)(b-d),2));
    }


    android.hardware.Camera.PictureCallback pictureCallback = new android.hardware.Camera.PictureCallback() {
        @Override
        public void onPictureTaken(byte[] data, android.hardware.Camera camera) {
            cameraSurface.stopPreview();
            Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
            bitmap = scaleDownBitmapImage(bitmap, previewSizeWidth, previewSizeHeight);
            //Matrix mtx = new Matrix();
            //mtx.postRotate(90);
            //bitmap = Bitmap.createBitmap(bitmap, 0, 0, previewSizeWidth, previewSizeHeight, mtx, true);
            int width = previewSizeWidth;
            int height = previewSizeHeight;

            nums_n = 0;
            nums_group_n = 0;
            devs_n = 0;
            int[] ns1 = JNIProcess.process(width, height, bitmap, pixels, bold_pixels, t_nums, nums_n, nums_group, nums_group_n,
                    t_devs, devs_n, Threshold);
            nums_n = ns1[0];
            nums_group_n = ns1[1];
            devs_n = ns1[2];
            //이미지처리 결과 출력
            Bitmap bitmapn;
            bitmapn = Bitmap.createBitmap(previewSizeWidth, previewSizeHeight, Bitmap.Config.ARGB_8888);
            int[] tmpPixels = new int[pixels.length/4];
            for(int i=0;i<tmpPixels.length;i++){
                tmpPixels[i] = (((pixels[i*4+3])*256+pixels[i*4+2])*256+pixels[i*4+1])*256+pixels[i*4];
            }
            bitmapn.setPixels(tmpPixels, 0, previewSizeWidth, 0, 0, previewSizeWidth, previewSizeHeight);
            Log.d("POE: OUT","OUT: "+nums_n+", "+nums_group_n+", "+devs_n);

            JNIProcess.matchinglabel(pixels, width, height, t_nums, nums_label, nums_n,
                    t_devs, devs_label, devs_bipolar, devs_n);

            //Numbering - JAVA
            for(int i=0;i<50;i++){
                number_group[i][0] = 0;
                number_group[i][1] = 0;
                number_group[i][2] = 0;
                number_group[i][3] = 0;
            }
            for(int i=0;i<nums_n;i++){
                number_group[nums_group[i]][0] = number_group[nums_group[i]][0] * 10 + nums_label[i];
                number_group[nums_group[i]][1] += (t_nums[i*4]+t_nums[i*4+2])/2; //center x
                number_group[nums_group[i]][2] += (t_nums[i*4+1]+t_nums[i*4+3])/2; //center y
                number_group[nums_group[i]][3] += 1; //n
            }
            //Number-Pict Matching - JAVA
            for(int i=0;i<devs_n;i++){
                circuit_devices[i][0] = devs_label[i];
                circuit_devices[i][1] = -1;
            }
            for(int i=1;i<=nums_group_n;i++){ //#1~nums_group_n
                int min = 0;
                for(int j=1;j<devs_n;j++){
                    int cxm, cxj, cym, cyj;
                    int dstm, dstj;
                    cxm = (t_devs[min*4+2] + t_devs[min*4])/2;
                    cym = (t_devs[min*4+3] + t_devs[min*4+1])/2;
                    cxj = (t_devs[j*4+2] + t_devs[j*4])/2;
                    cyj = (t_devs[j*4+3] + t_devs[j*4+1])/2;
                    dstm = (number_group[i][1]/number_group[i][3]-cxm)*(number_group[i][1]/number_group[i][3]-cxm)
                            + (number_group[i][2]/number_group[i][3]-cym)*(number_group[i][2]/number_group[i][3]-cym);
                    dstj = (number_group[i][1]/number_group[i][3]-cxj)*(number_group[i][1]/number_group[i][3]-cxj)
                            + (number_group[i][2]/number_group[i][3]-cyj)*(number_group[i][2]/number_group[i][3]-cyj);
                    if(dstm>dstj && dstj>=200){ //일정거리 이하만 매칭
                        min = j;
                    }
                }
                int cxm, cym;
                int dstm; //일정거리 이하인 숫자.기호 매칭
                cxm = (t_devs[min*4+2] + t_devs[min*4])/2;
                cym = (t_devs[min*4+3] + t_devs[min*4+1])/2;
                dstm = (number_group[i][1]/number_group[i][3]-cxm)*(number_group[i][1]/number_group[i][3]-cxm)
                        + (number_group[i][2]/number_group[i][3]-cym)*(number_group[i][2]/number_group[i][3]-cym);
                if(dstm>=200) circuit_devices[min][1] = number_group[i][0];
            }

            //int[] pixels_p = new int[width*height];
            //for(int i=0;i<height*width;i++) pixels_p[i] = bold_pixels[i*4];
            analysis_node_n = 0;
            for(int i=0;i<devs_n;i++){
                circuit_devices[i][2] = -1;
                circuit_devices[i][3] = -1;
            }
            for(int ii=0;ii<devs_n;ii++){ //blur 4에 대한 보상, 번져서 범위 외부에도 흰점 발생하였기 때문에 방지
                t_devs[ii*4] -= 3;
                t_devs[ii*4+1] -= 3;
                t_devs[ii*4+2] += 3;
                t_devs[ii*4+3] += 3;
            }
            for(int ii=0;ii<height;ii++){ //노드 구성
                for(int jj=0;jj<width;jj++){
                    if((bold_pixels[(ii*width+jj)]&0xFF)!=255) continue;
                    int i;
                    for(i=0;i<devs_n;i++){ //device에서 출발하지 않기
                        if((t_devs[i*4])<=jj && jj<=(t_devs[i*4+2]) && (t_devs[i*4+1])<=ii && ii<=(t_devs[i*4+3])){
                            break;
                        }
                    }
                    if(i!=devs_n) continue;
                    int[][] queue = new int[2000][2]; int front=0; int end=0;
                    int[] vertex = new int[50]; //해당 노드에 연결된 device
                    int vn = 0;
                    for(i=0;i<devs_n;i++)
                        vertex[i]=0;
                    queue[end][0] = jj; queue[end][1] = ii; end++;
                    while(front!=end){
                        int px = queue[front][0];
                        int py = queue[front][1];
                        front = (front+1)%2000;
                        if(!(0<=px && px<width && 0<=py && py<height)) continue;
                        if((bold_pixels[(py)*width+(px)]&0xFF) != 255) continue;
                        bold_pixels[py*width+px] = 8;
                        for(i=0;i<devs_n;i++){
                            if((t_devs[i*4])<=px && px<=(t_devs[i*4+2]) && (t_devs[i*4+1])<=py && py<=(t_devs[i*4+3])){
                                if(vertex[i]==0){
                                    vn++;
                                    if(dist(px, py, devs_bipolar[i*4+0], devs_bipolar[i*4+1]) <= dist(px, py, devs_bipolar[i*4+2], devs_bipolar[i*4+3])) vertex[i] = 1;
                                    else vertex[i] = 2;
                                    //printf("%d %d: %d %d %d %lf %lf %lf %lf\n", i, analysis_node_n, vertex[i], px, py, devs_bipolar[i*4+0], devs_bipolar[i*4+1], devs_bipolar[i*4+2], devs_bipolar[i*4+3]);
                                }
                                break;
                            }
                        }
                        if(i!=devs_n) continue;
                        queue[end][0] = px+1; queue[end][1] = py; end=(end+1)%2000;
                        queue[end][0] = px-1; queue[end][1] = py; end=(end+1)%2000;
                        queue[end][0] = px; queue[end][1] = py+1; end=(end+1)%2000;
                        queue[end][0] = px; queue[end][1] = py-1; end=(end+1)%2000;
                    }
                    if(vn>1){
                        for(i=0;i<devs_n;i++){
                            if(vertex[i]!=0){
                                if(circuit_devices[i][0]==0){
                                    if(circuit_devices[i][2]==-1) circuit_devices[i][2] = analysis_node_n;
                                    else if(circuit_devices[i][3]==-1) circuit_devices[i][3] = analysis_node_n;
                                    else Log.d("POE","node error");
                                }
                                else{
                                    //printf("%d %d: %d %d %d\n", i, analysis_node_n, vertex[i], circuit_devices[i][2], circuit_devices[i][3]);
                                    if(vertex[i]==1 && circuit_devices[i][2]==-1) circuit_devices[i][2] = analysis_node_n;
                                    else if(vertex[i]==2 && circuit_devices[i][3]==-1) circuit_devices[i][3] = analysis_node_n;
                                    else Log.d("POE","node error");
                                }
                            }
                        }
                        analysis_node_n++;
                    }
                }
            }
            Log.d("POE", analysis_node_n+".");
            for(int i=0;i<devs_n;i++){
                if(circuit_devices[i][2]<0 || circuit_devices[i][3]<0){
                    resultText.setText("Circuit Error");
                    imgView.setVisibility(View.VISIBLE);
                    imgView.SetBitmap(bitmapn);
                    imgView.invalidate();
                    return;
                }
            }

            int devs_n_tmp = devs_n;
            for(int i=0;i<devs_n_tmp;i++){
                circuit_constant[i] = (double)circuit_devices[i][1];
                if(circuit_devices[i][0]==1){ //전압원 -> 전류원
                    circuit_constant[i] = circuit_constant[i] / 0.000001;
                    circuit_devices[devs_n][0] = 0; //병렬 저항 추가
                    circuit_devices[devs_n][1] = 0;
                    circuit_devices[devs_n][2] = circuit_devices[i][2];
                    circuit_devices[devs_n][3] = circuit_devices[i][3];
                    circuit_constant[devs_n] = 0.000001;
                    t_devs[devs_n*4] = -1;
                    t_devs[devs_n*4+1] = -1;
                    devs_n++;
                }
                else if(circuit_devices[i][0]==2){ // 전압계 저항 inf
                    circuit_constant[i] = 1000000;
                }
            }
            for(int i = 0; i<devs_n;i++) {
                if (circuit_devices[i][0] == 1) { //전압계 입력
                    circuit_constant[i] = Voltage / 0.000001;
                }
                Log.d("POE",i+":"+circuit_devices[i][0]+" "+circuit_constant[i]+" "+circuit_devices[i][2]+" "+circuit_devices[i][3]);
            }
            computeNode(circuit_devices, circuit_constant, devs_n, analysis_result, analysis_node_n);

            for(int i=0;i<devs_n;i++){
                if(circuit_devices[i][0]==2){ //전압계 출력
                    Log.d("POE","OUTPUT: "+ (analysis_result[circuit_devices[i][3]]-analysis_result[circuit_devices[i][2]]));
                    //    printf("OUTPUT: %lf\n", analysis_result[circuit_devices[i][3]]-analysis_result[circuit_devices[i][2]]);
                    String output_v = String.format("%.3f", (analysis_result[circuit_devices[i][3]]-analysis_result[circuit_devices[i][2]]));
                    String str = "OUTPUT: "+output_v+" [V]";
                    resultText.setText(str);
                }
            }


            /*Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
            int w = bitmap.getWidth();
            int h = bitmap.getHeight();
            bitmap = scaleDownBitmapImage(bitmap, 800, 700);
            Matrix mtx = new Matrix();
            mtx.postRotate(180);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, w, h, mtx, true);

            if (bitmap == null) {
                Toast.makeText(MainActivity.this, "Captured image is empty", Toast.LENGTH_SHORT).show();
                return;
            }*/
            imgView.setVisibility(View.VISIBLE);
            imgView.SetBitmap(bitmapn);
            imgView.invalidate();
        }
    };

    private void computeNode(int[][] circuit_devices, double[] circuit_constant, int devs_n, double[] analysis_result, int analysis_node_n)
    {
        analysis_result[analysis_node_n-1] = 0; //맨 마지막노드 reference node
        double[][] matrix = new double[100][100]; //node 계산행렬
        for(int i=0;i<analysis_node_n;i++){
            analysis_result[i] = 0;
            for(int j=0;j<analysis_node_n;j++){
                matrix[i][j] = 0;
            }
        }
        for(int i=0;i<analysis_node_n-1;i++){ //nodal analysis
            for(int k = 0; k < devs_n; k++){
                if(circuit_devices[k][0]==0 || circuit_devices[k][0]==2){ //저항 or 전압계
                    if(circuit_devices[k][2]==i){
                        matrix[i][circuit_devices[k][2]] += (double)1.0/circuit_constant[k];
                        matrix[i][circuit_devices[k][3]] -= (double)1.0/circuit_constant[k];
                    }else if(circuit_devices[k][3]==i){
                        matrix[i][circuit_devices[k][3]] += (double)1.0/circuit_constant[k];
                        matrix[i][circuit_devices[k][2]] -= (double)1.0/circuit_constant[k];
                    }
                }
                else if(circuit_devices[k][0]==1){ //전류원
                    if(circuit_devices[k][2]==i){
                        analysis_result[i] -= circuit_constant[k];
                    }else if(circuit_devices[k][3]==i){
                        analysis_result[i] += circuit_constant[k];
                    }
                }
            }
        }
        //Gauss Eli
        for(int i=0;i<analysis_node_n-1;i++){
            for(int j=i+1;j<analysis_node_n-1;j++){
                for(int k=i;k<analysis_node_n-1;k++){
                    matrix[i][k] += matrix[j][k];
                }
                analysis_result[i] += analysis_result[j];
            }

            double r = matrix[i][i];
            for(int k=i;k<analysis_node_n-1;k++){
                matrix[i][k] = matrix[i][k] / r;
            }
            analysis_result[i] = analysis_result[i] / r;

            for(int j=i+1;j<analysis_node_n-1;j++){
                double rn = matrix[j][i];
                for(int k=i;k<analysis_node_n-1;k++){
                    matrix[j][k] -= matrix[i][k] * rn;
                }
                analysis_result[j] -= analysis_result[i] * rn;
            }
        }
        //Jordan Eli
        for(int j=analysis_node_n-1; j>=1;j--){
            for(int i=0; i<j; i++){
                double rn = matrix[i][j];
                matrix[i][j] -= matrix[j][j] * rn;
                analysis_result[i] -= analysis_result[j] * rn;
            }
        }

    }

    private Bitmap scaleDownBitmapImage(Bitmap bitmap, int newWidth, int newHeight){
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
        return resizedBitmap;
    }

}
