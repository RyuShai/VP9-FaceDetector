package com.example.ryu.facedetectpreview;

import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ServiceConnection;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Environment;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
  import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.sql.Date;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Set;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.resize;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {
    /*
    * Notifications from UsbService will be received here.
    */
    private final BroadcastReceiver mUsbReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            switch (intent.getAction()) {
                case UsbService.ACTION_USB_PERMISSION_GRANTED: // USB PERMISSION GRANTED
                    Toast.makeText(context, "USB Ready", Toast.LENGTH_SHORT).show();
                    break;
                case UsbService.ACTION_USB_PERMISSION_NOT_GRANTED: // USB PERMISSION NOT GRANTED
                    Toast.makeText(context, "USB Permission not granted", Toast.LENGTH_SHORT).show();
                    break;
                case UsbService.ACTION_NO_USB: // NO USB CONNECTED
                    Toast.makeText(context, "No USB connected", Toast.LENGTH_SHORT).show();
                    break;
                case UsbService.ACTION_USB_DISCONNECTED: // USB DISCONNECTED
                    Toast.makeText(context, "USB disconnected", Toast.LENGTH_SHORT).show();
                    break;
                case UsbService.ACTION_USB_NOT_SUPPORTED: // USB NOT SUPPORTED
                    Toast.makeText(context, "USB device not supported", Toast.LENGTH_SHORT).show();
                    break;
            }
        }
    };

    @Override
    public void onClick(View view) {
        switch (view.getId())
        {
            case R.id.btnTakePicture:
            {
                break;
            }
        }
    }

    public enum DIRECTION{
        UP,
        DOWN,
        LEFT,
        RIGHT
    }
    private static String TAG = "Ryu_MainActivity";
    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadesClassifier;
    private CascadeClassifier foreheadCascade;
    private Mat grayScaleImage;
    private int absoluteFaceSize;
    private TextView display;
    Button btnUp,btnDown,btnLeft,btnRight,btnCenter,btnTakePicure;
    Spinner listFolder;
    JavaCameraView jCamera;

    //test
    Mat foreheadFrame;
    MatOfRect foreheadList;
    List<Mat> history_frames;
    List<MatOfRect> history_faces;
    float scale =3.0f;
    TextView txtView;
    //
    private boolean takePicture=false;
    private List<String> folders;
    private Spinner spinner;
    private int  valueX=0,valueY=0,step=1,stepMax=10,stepMin=1;
    private UsbService usbService;
    private MyHandler mHandler;
    private final ServiceConnection usbConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName arg0, IBinder arg1) {
            usbService = ((UsbService.UsbBinder) arg1).getService();
            usbService.setHandler(mHandler);
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            usbService = null;
        }
    };


    BaseLoaderCallback mLoaderCallBack= new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            switch (status)
            {
                case BaseLoaderCallback.SUCCESS:{
                    initializeOpenCVDependencies();
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    Mat mRgba;
    //
    // Used to load the 'native-lib' library on application startup.

    static{
        if(OpenCVLoader.initDebug())
        {
            Log.d(TAG,"opencv loadded");
        }else{
            Log.d(TAG,"opencv load failed");

        }
    }


    //function
    private void initializeOpenCVDependencies(){
        try{
            InputStream is = getResources().openRawResource(R.raw.cascade_tuc2);
            File cascadeDir  = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File (cascadeDir, "haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while((bytesRead = is.read(buffer)) != -1){
                os.write(buffer,0,bytesRead);
            }
            is.close();
            os.close();
            foreheadCascade = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        }catch(Exception ex)
        {
            Log.e(TAG,"error loadding cascade", ex);
        }

        openCvCameraView.enableView();
    }
    //

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);

//Remove notification bar
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_main);
        //init gui
        initButton();
        String path = Environment.getExternalStorageDirectory()+"/Ryu";
        Log.d(TAG,path);
        GetListFolder(path);

//        Camera camera = Camera.open();
//        Camera.Parameters params = camera.getParameters();
//        List<Camera.Size> sizes = params.getSupportedPictureSizes();
//        for (Camera.Size cam:
//             sizes) {
//            txtView.setText(txtView.getText()+ String.valueOf(cam.width) + "-"+String.valueOf(cam.height)+"\n");
//        }
//        camera.release();

        openCvCameraView = (CameraBridgeViewBase) findViewById(R.id.Camera);
        openCvCameraView.setVisibility(SurfaceView.VISIBLE);
        openCvCameraView.setCvCameraViewListener(this);
        //handle serial
        mHandler = new MyHandler(this);
        //
        foreheadFrame = new Mat();
        foreheadList = new MatOfRect();
        initializeOpenCVDependencies();
        history_frames = new ArrayList<Mat>();
        history_faces = new ArrayList<MatOfRect>();
        foreheadFrame = new Mat();
        //last option
//        try {
//            FileInputStream fis = new FileInputStream(yamlPath);
//            BufferedReader bfr = new BufferedReader(new InputStreamReader(fis));
//            String line = bfr.readLine();
//            while(line!=null)
//            {
//                Log.d("yaml", line);
//                line = bfr.readLine();
//            }
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(jCamera!=null)
            jCamera.disableView();
        unregisterReceiver(mUsbReceiver);
        unbindService(usbConnection);
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(jCamera!=null)
            jCamera.disableView();
    }

    @Override
    protected void onResume(){
        super.onResume();
        setFilters();  // Start listening notifications from UsbService
        startService(UsbService.class,usbConnection,null); // Start UsbService(if it was not started before) and Bind it
        if(OpenCVLoader.initDebug())
        {
            Log.d(TAG,"opencv loadded");
            mLoaderCallBack.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }else{
            Log.d(TAG,"opencv load failed");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, MainActivity.this , mLoaderCallBack);
        }

    }
    void detectCPU(double scale, boolean calTime)
    {
        Mat cpu_gray = new Mat();
        Mat cpu_smallImg= new Mat( Round(foreheadFrame.rows()/scale), Round(foreheadFrame.cols()/scale), CV_8UC1);
        Imgproc.cvtColor(foreheadFrame, cpu_gray, Imgproc.COLOR_RGB2GRAY);
        resize(cpu_gray, cpu_smallImg, cpu_smallImg.size(), 0, 0, INTER_LINEAR);
        Log.e(TAG, "size: "+cpu_smallImg.size());
        Imgproc.equalizeHist(cpu_smallImg, cpu_smallImg);
        foreheadCascade.detectMultiScale(cpu_smallImg, foreheadList, 1.1, 5, Objdetect.CASCADE_SCALE_IMAGE,
               new Size(60, 60),new Size(160, 160)); //Size(40, 40), Size(70, 70));

    }

    int Round(double x){
        int y;
        if(x >= (int)x+0.5)
        y = (int)x++;
        else
        y = (int)x;
        return y;
    }
    //init widget gui
    private void initButton()
    {
        btnCenter = (Button) findViewById(R.id.btnCenter);
        btnUp = (Button) findViewById(R.id.btnUp);
        btnDown =(Button) findViewById(R.id.btnDown);
        btnLeft = (Button) findViewById(R.id.btnLeft);
        btnRight = (Button) findViewById(R.id.btnRight);
        btnTakePicure = (Button) findViewById(R.id.btnTakePicture);
        txtView = (TextView) findViewById(R.id.textView);

        btnCenter.setText(String.valueOf(step));
        btnCenter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                step++;
                if(step>stepMax) step = stepMin;
                btnCenter.setText(String.valueOf(step));
            }
        });

        btnUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                DirectionBtnClicked(DIRECTION.UP);
            }
        });

        btnDown.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                DirectionBtnClicked(DIRECTION.DOWN);
            }
        });

        btnLeft.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                DirectionBtnClicked(DIRECTION.LEFT);
            }
        });

        btnRight.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                DirectionBtnClicked(DIRECTION.RIGHT);
            }
        });

        btnTakePicure.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                takePicture = true;
            }
        });
    }

    void GetListFolder(String path)
    {
        folders = new ArrayList<String>();
        File f = new File(path);
        File[] files = f.listFiles();
        for (File inFile : files) {
            if (inFile.isDirectory()) {
                // is directory
                Log.d(TAG,inFile.getAbsolutePath());
                folders.add(inFile.getAbsolutePath());
            }
        }
        spinner = (Spinner) findViewById(R.id.spinner);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(
                this,android.R.layout.simple_dropdown_item_1line,folders
        );
        spinner.setAdapter(adapter);
    }

    void DirectionBtnClicked(DIRECTION direction)
    {
        switch (direction)
        {
            case UP:
                valueX+=step;
                sendData(valueX,valueY);
                break;
            case DOWN:
                valueX-=step;
                sendData(valueX,valueY);
                break;
            case LEFT:
                valueY-=step;
                sendData(valueX,valueY);
                break;
            case RIGHT:
                valueY+=step;
                sendData(valueX,valueY);
                break;
        }
    }

    void sendData(int posX, int posY)
    {
        if(posX<0) posX=0;
        if(posX>4000) posX=4000;
        if(posY<0) posY=0;
        if(posY>4000) posY=4000;

        String data = "X"+String.valueOf(posX)+" Y"+String.valueOf(posY)+" ";
        Toast.makeText(getApplicationContext(),data,Toast.LENGTH_SHORT).show();
        usbService.write(data.getBytes());
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    @Override
    public void onCameraViewStarted(int width, int height) {
       grayScaleImage = new Mat(1080,1920,CvType.CV_8UC3);

//        absoluteFaceSize = (int) (height*0.2);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        foreheadFrame.release();
        inputFrame.rgba().copyTo(foreheadFrame);

        Log.e(TAG, foreheadFrame.size() + " " + foreheadFrame.empty());

        detectCPU(scale,false);
        history_frames.add(foreheadFrame);
        history_faces.add(foreheadList);
        int cur_size = history_frames.size();
        if(cur_size>4)
        {
//            double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
            //std::cout << "FPS : " << fps << std::endl;

            //Using this for DNN filter out some non-head Mat
            List<Boolean> yesheads = new ArrayList<Boolean>();


            for (int i=0; i<(history_faces.get(cur_size-3).toList().size());i++) yesheads.add(true);


            //------Do the final check to remove moise/////////////////////////////////////////////////

            //Check the face list with previous and later frame, make sure it is face, not noise
            MatOfRect faces_check1, faces_check2;
            faces_check1 = history_faces.get(cur_size-3);
            faces_check2 = history_faces.get(cur_size-4);
            //Check face between consecutive frames to remove noise
            for (int i=0;i<faces_check1.toList().size();i++){
                Point pt1 = new Point(faces_check1.toList().get(i).x+faces_check1.toList().get(i).width/2, faces_check1.toList().get(i).y+faces_check1.toList().get(i).height/2);
                float mindist = 100000;
                for (int j=0;j<faces_check2.toList().size();j++){
                    Point pt2 = new Point(faces_check2.toList().get(j).x+faces_check2.toList().get(j).width/2, faces_check2.toList().get(j).y+faces_check2.toList().get(j).height/2);
                    float dist = (faces_check1.toList().get(i).x-faces_check2.toList().get(j).x)*(faces_check1.toList().get(i).x-faces_check2.toList().get(j).x)+
                            (faces_check1.toList().get(i).y-faces_check2.toList().get(j).y)*(faces_check1.toList().get(i).y-faces_check2.toList().get(j).y);
                    if (dist<mindist) mindist = dist;
                }
                if (mindist>50) yesheads.set(i,false);
            }

            //Check face in previewing frame to remove occlusion
            for (int i=0;i<faces_check1.toList().size();i++){
                if (yesheads.get(i)==false) continue;
                for (int j=i+1;j<faces_check1.toList().size();j++){
                    if (yesheads.get(j)==false) continue;
                    float distx = Math.abs(faces_check1.toList().get(i).x-faces_check1.toList().get(j).x);
                    float disty = Math.abs(faces_check1.toList().get(i).y-faces_check1.toList().get(j).y);
                    if (distx<(int) faces_check1.toList().get(i).width && disty<5*(int) faces_check1.toList().get(i).height){
                        if (faces_check1.toList().get(i).y>faces_check1.toList().get(j).y) yesheads.set(j,false);
                        else yesheads.set(i,false);
                    }
                }
            }

            ///////////////////////NOW DRAW FRAME/////////////////////////////////////////////
            int i = 0;
            int counthead =0;
            Mat img = history_frames.get(cur_size-3);
            for( ; i<faces_check1.toList().size(); i++ )
            {
                if (yesheads.get(i)){
                    Rect rect = faces_check1.toArray()[i];
                    // std::cout<<faces_check1[i].y<<" "<<(int) img.rows/3<<std::endl;
                    if (faces_check1.toArray()[i].y >(int) img.rows()/3 && faces_check1.toArray()[i].y<(int) img.rows()*2/3)
                    {
                        double p1 = faces_check1.toArray()[i].x+(int) faces_check1.toArray()[i].width/5;
                        double p2 = faces_check1.toArray()[i].y+(int) faces_check1.toArray()[i].height/2.7;
                        double p3 =  faces_check1.toArray()[i].width*3/5;
                        double p4 = faces_check1.toArray()[i].height/3;
                        Imgproc.rectangle(img,new Point(p1,p2), new Point(p3,p4),new Scalar(0,255,0),2);
                    }
                        //cv::rectangle(img,faces_check1[i],CV_RGB(0,255,0),2);

                    //Check here if value is negative
                    counthead++;
                }
            }
            if( Math.abs(scale-1.0)>.001 )
            {
                resize(img, img,new Size((int)(img.cols()/scale), (int)(img.rows()/scale)));
            }
            //string str = "Number of people: "+std::to_string(counthead);
            //putText(img, str, Point(20,20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,0), 2.0 );
            takePicture=false;
            history_faces.clear();
            history_frames.clear();
            return img;
        }

        if(takePicture)
        {
            imagePrint(inputFrame.rgba());
            takePicture=false;
        }
        return inputFrame.rgba();
    }


    public void imagePrint(Mat mat)
    {
        int iNum =0;
        Bitmap bmp =null;
        try{
            bmp = Bitmap.createBitmap(mat.cols(),mat.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat,bmp);
        }catch (CvException e){
            Log.e(TAG,"convert to bitmap failed");
        }
        FileOutputStream os= null;
        String fileName ;
        File sd = new File(spinner.getSelectedItem().toString());
        boolean success = true;
        if(!sd.exists()){
            success = sd.mkdir();
        }
        if(success){
            File dest;
            do{
                java.util.Date current = Calendar.getInstance().getTime();
                fileName = sd.getAbsolutePath()+"/"+current.toString()+".png";
                Log.d(TAG,"file path: "+ fileName);
                dest= new File(fileName);
            }while(dest.exists());
            try{
                os = new FileOutputStream(dest);
                bmp.compress(Bitmap.CompressFormat.PNG,100,os);
            }catch (Exception e)
            {
                Log.e(TAG,"write image file : ",e);
            }
            finally {
                {
                    try{
                        if(os!=null){
                            os.close();
                            Log.d(TAG,"OK");
                        }
                    }catch (IOException e){
                        Log.e(TAG,"error: "+e.getMessage());
                    }
                }
            }
        }
    }

    //process write image wil write image to screen in other thread
    private class ProcessWriteImage extends AsyncTask<Void,Void,String>{
        MainActivity mainAct;
        public ProcessWriteImage(MainActivity main){
            this.mainAct = main;
        }

        @Override
        protected void onPostExecute(String s) {
            super.onPostExecute(s);
        }

        @Override
        protected String doInBackground(Void... params) {
            mainAct.imagePrint(mainAct.mRgba);
            return null;
        }
    }

    //create service
    private void startService(Class<?> service, ServiceConnection serviceConnection, Bundle extras) {
        if (!UsbService.SERVICE_CONNECTED) {
            Intent startService = new Intent(this, service);
            if (extras != null && !extras.isEmpty()) {
                Set<String> keys = extras.keySet();
                for (String key : keys) {
                    String extra = extras.getString(key);
                    startService.putExtra(key, extra);
                }
            }
            startService(startService);
        }
        Intent bindingIntent = new Intent(this, service);
        bindService(bindingIntent, serviceConnection, Context.BIND_AUTO_CREATE);
    }

    private void setFilters() {
        IntentFilter filter = new IntentFilter();
        filter.addAction(UsbService.ACTION_USB_PERMISSION_GRANTED);
        filter.addAction(UsbService.ACTION_NO_USB);
        filter.addAction(UsbService.ACTION_USB_DISCONNECTED);
        filter.addAction(UsbService.ACTION_USB_NOT_SUPPORTED);
        filter.addAction(UsbService.ACTION_USB_PERMISSION_NOT_GRANTED);
        registerReceiver(mUsbReceiver, filter);
    }
    /*
     * This handler will be passed to UsbService. Data received from serial port is displayed through this handler
     */
    private static class MyHandler extends Handler {
        private final WeakReference<MainActivity> mActivity;

        public MyHandler(MainActivity activity) {
            mActivity = new WeakReference<>(activity);
        }

        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case UsbService.MESSAGE_FROM_SERIAL_PORT:
//                    String data = (String) msg.obj;
//                    mActivity.get().display.append(data);
                    break;
                case UsbService.CTS_CHANGE:
                    Toast.makeText(mActivity.get(), "CTS_CHANGE",Toast.LENGTH_LONG).show();
                    break;
                case UsbService.DSR_CHANGE:
                    Toast.makeText(mActivity.get(), "DSR_CHANGE",Toast.LENGTH_LONG).show();
                    break;
            }
        }
    }
}


