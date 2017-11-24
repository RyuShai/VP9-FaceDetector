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
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
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
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.ref.WeakReference;
import java.security.cert.PolicyNode;
import java.sql.Date;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Set;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.cvtColor;
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
    Rect subRect ;
    Mat foreheadFrame ;
    MatOfRect foreheadList;
    List<Mat> history_frames;
    List<MatOfRect> history_faces;
    float scale =3.0f;
    TextView txtView;
    Rect result;
    List<Integer> resultCali;
    //
    boolean StartDetect=false;
    LaserColor colorMean;
    List<LaserColor> lColor;
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
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
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
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        //init gui
        initButton();
        String path = Environment.getExternalStorageDirectory()+"/Ryu";
        Log.d(TAG,path);
        GetListFolder(path);
        Mat testMat = readFromFile(Environment.getExternalStorageDirectory()+"/cam2laserMatrices.yml");

//        Camera camera = Camera.open();
//        Camera.Parameters params = camera.getParameters();
//        List<Camera.Size> sizes = params.getSupportedPictureSizes();
//        for (Camera.Size cam:
//             sizes) {
////            txtView.setText(txtView.getText()+ String.valueOf(cam.width) + "-"+String.valueOf(cam.height)+"\n");
//            Log.e(TAG,String.valueOf(cam.width) + "-"+String.valueOf(cam.height));
//        }
//        camera.release();
        initCamera();
        colorMean = new LaserColor();
        subRect = new Rect(460,400,530,350);// khu vuc detect
        foreheadFrame = new Mat();
        foreheadList = new MatOfRect();
        lColor = new ArrayList<LaserColor>();
        //handle serial
        mHandler = new MyHandler(this);
        //
        initializeOpenCVDependencies();
        history_frames = new ArrayList<Mat>();
        history_faces = new ArrayList<MatOfRect>();
        resultCali = new ArrayList<Integer>();
        result = new Rect();
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
//    void detectCPU(double scale, boolean calTime)
//    {
//        Mat cpu_gray = new Mat();
//        Mat cpu_smallImg= new Mat( Round(foreheadFrame.rows()/scale), Round(foreheadFrame.cols()/scale), CV_8UC1);
//        Imgproc.cvtColor(foreheadFrame, cpu_gray, Imgproc.COLOR_RGB2GRAY);
//        resize(cpu_gray, cpu_smallImg, cpu_smallImg.size(), 0, 0, INTER_LINEAR);
//        Log.e(TAG, "size: "+cpu_smallImg.size());
//        Imgproc.equalizeHist(cpu_smallImg, cpu_smallImg);
//        foreheadCascade.detectMultiScale(cpu_smallImg, foreheadList, 1.1, 5, Objdetect.CASCADE_SCALE_IMAGE,
//               new Size(60, 60),new Size(160, 160)); //Size(40, 40), Size(70, 70));
//
//    }
    int iMeanCount=0;
    void initCamera()
    {
        openCvCameraView = (CameraBridgeViewBase) findViewById(R.id.Camera);
        openCvCameraView.setMaxFrameSize(1920,1080);
        openCvCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if(motionEvent.getAction() == MotionEvent.ACTION_DOWN && iMeanCount <9 )
                {
                    int x = (int) motionEvent.getX();
                    int y = (int) motionEvent.getY();
                    txtView.setText("X: "+x + " Y "+y);
                    Mat getColorMat = foreheadFrame.clone();
                    Imgproc.cvtColor(getColorMat,getColorMat,Imgproc.COLOR_RGBA2RGB);
                    double[] color = getColorMat.get(x,y);
                    LaserColor addColor = new LaserColor((int)color[0],(int)color[1],(int)color[2]);
                    lColor.add(addColor);
                    iMeanCount++;
                    if(iMeanCount==9)
                    {
                        for(int i=0; i<lColor.size();i++)
                        {
                            colorMean.setBlue(colorMean.getBlue()+lColor.get(i).getBlue());
                            colorMean.setGreen(colorMean.getGreen()+lColor.get(i).getGreen());
                            colorMean.setRed(colorMean.getRed()+lColor.get(i).getRed());
                        }
                        colorMean.setBlue(colorMean.getBlue()/9);
                        colorMean.setGreen(colorMean.getGreen()/9);
                        colorMean.setRed(colorMean.getRed()/9);
                        DebugLog.isStartDetect = true;
                        DebugLog.meanColorValue = colorMean.rgbString();
                    }
                }

                return false;
            }
        });
        openCvCameraView.setVisibility(SurfaceView.VISIBLE);
        openCvCameraView.setCvCameraViewListener(this);
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

    //create mat from yaml
    Mat readFromFile(String path)
    {
        Mat matReturn ;
        try {
            File file = new File(path);
            BufferedReader br = new BufferedReader(new FileReader(file));
            String line = br.readLine();

            int row=0, col=0;
            while(line!=null)
            {
                Log.e(TAG,line);
                if(line.contains("rows:"))
                {
                    String[] strArray = line.split("\\:",2);
                    row = Integer.parseInt(strArray[1] .trim());
                    line = br.readLine();
                }
                if(line.contains("cols:"))
                {
                    String[] strArray = line.split("\\:",2);
                    col = Integer.parseInt(strArray[1].trim());
                    line = br.readLine();
                }
                if(row>0 && col>0)
                {
                    matReturn = new Mat(row,col, CvType.CV_32FC1);
                    line = br.readLine();
                    String matValue="";

                    if(line.contains("["))
                    {
                        line = line.split("\\:",2)[1];
                        while(!line.contains("]"))
                        {
                            matValue+=line;
                            line = br.readLine();
                        }
                        matValue +=line;
                        Log.e(TAG,matValue);
                        matValue=matValue.replace("[","");
                        matValue=matValue.replace("]","");
                        String[] arrString = matValue.split(",");
                        if(arrString.length!= (row*col))
                        {
                            Log.e(TAG,"loi me no roi ");
                        }
                        else
                        {
                            double[] matByte = new double[row*col];
                            for( int i=0; i<matByte.length;i++)
                            {
                                matByte[i] = Float.parseFloat(arrString[i].trim());
                                Log.e(TAG,"Value:" + matByte[i]);
                            }
                            matReturn.put(0,0,matByte);
                            return matReturn;
                        }
                    }
                }
                line = br.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    void GetListFolder(String path)
    {
        txtView.setText(path);
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
//        absoluteFaceSize = (int) (height*0.2);
    }

    @Override
    public void onCameraViewStopped() {

    }
    double start;
    List<Rect> list5Rect = new ArrayList<Rect>();
    Point center = new Point();
    Rect getFitRect(MatOfRect listRect)
    {
        List<Mat> listMat = new ArrayList<Mat>();
        Rect rectReturn = new Rect();
        int deviation = 195075;
        Mat smallMat ;
        for(int i=0; i<listRect.toArray().length;i++)
        {
            Rect currentRect = listRect.toArray()[i];
            int x =currentRect.x;
            int y = currentRect.y+(int)Math.floor(currentRect.height/3*2);
            int w = currentRect.width/3*2;
            int h =(int)Math.floor(currentRect.height/3/2);
            Log.e(TAG, String.valueOf(y)+" "+String.valueOf(y+h)+" "+String.valueOf(x)+" "+String.valueOf(x+w)+" "+foreheadFrame.size().toString());
            smallMat=foreheadFrame.submat(y,y+h,x,x+w);
            LaserColor color = new LaserColor();
            for(int row=0;row<smallMat.rows();row++)
            {
                for(int col=0; col<smallMat.cols();col++)
                {
                    color.setRed(color.getRed()+(int)smallMat.get(row,col)[0]);
                    color.setGreen(color.getGreen()+(int)smallMat.get(row,col)[1]);
                    color.setBlue(color.getBlue()+(int)smallMat.get(row,col)[2]);
                }
            }
            int total = smallMat.rows()*smallMat.cols();
            color.setBlue(color.getBlue()/total);
            color.setGreen(color.getGreen()/total);
            color.setRed(color.getRed()/total);
            DebugLog.currentColorValue = String.valueOf(color.getBlue()/total);
            int currentDeviation =(int)(Math.pow((colorMean.getRed()-color.getRed()),2) + Math.pow((colorMean.getGreen()-color.getGreen()),2) + Math.pow((colorMean.getBlue()-color.getBlue()),2));
            if(currentDeviation < deviation)
            {
                deviation = currentDeviation;
                list5Rect.add(currentRect);
                if(list5Rect.size()>10)
                    list5Rect.remove(0);

                for(int iCount=0; iCount<list5Rect.size();iCount++)
                {
                    rectReturn.x+=list5Rect.get(iCount).x;
                    rectReturn.y+=list5Rect.get(iCount).y;
                    rectReturn.width+=list5Rect.get(iCount).width;
                    rectReturn.height+=list5Rect.get(iCount).height;
                    center.x +=x+w/2;
                    center.y +=y+h/2;
                }
                rectReturn.x=Math.round(rectReturn.x/list5Rect.size());
                rectReturn.y=Math.round(rectReturn.y/list5Rect.size());
                rectReturn.width=Math.round(rectReturn.width/list5Rect.size());
                rectReturn.height=Math.round(rectReturn.height/list5Rect.size());
            }

        }
//        for(int i=0; i<listMat.size();i++)
//        {
//            imagePrint(listMat.get(i));
//        }
        return rectReturn;
    }
    @Override
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        Log.e(TAG,"start camera frame");
         start = System.currentTimeMillis();
        inputFrame.rgba().copyTo(foreheadFrame);
        if(!DebugLog.isStartDetect)
            return foreheadFrame;
//        Log.e(TAG,String.valueOf(foreheadFrame.width()+" "+String.valueOf(foreheadFrame.height())));
        if(foreheadFrame.width()==1920 && foreheadFrame.height()==1080)
        {
            DetectProcess();
        }
        Imgproc.rectangle(foreheadFrame,subRect.tl(),subRect.br(),new Scalar(0,255,0),2);

        if(takePicture)
        {
            imagePrint(foreheadFrame);
            takePicture=false;
        }

        haFunction();
        
        DebugLog.processTime = String.valueOf(System.currentTimeMillis()-start);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                txtView.setText(DebugLog.printLog());
            }
        });

        return foreheadFrame;
    }

    void DetectProcess()
    {
        Mat subMat = new Mat(foreheadFrame,subRect);
        cvtColor(subMat,subMat,Imgproc.COLOR_RGB2GRAY);
        Imgproc.resize(subMat,subMat,new Size(subMat.width()/2,subMat.height()/2));

        foreheadCascade.detectMultiScale(subMat, foreheadList, 1.1, 3, Objdetect.CASCADE_SCALE_IMAGE,
                new Size(50, 30),new Size(160, 100)); //Size(40, 40), Size(70, 70));

        Rect drawRect = new Rect();

        if(foreheadList.toArray().length>1)
        {
            drawRect=getFitRect(foreheadList);
        }
        else if(foreheadList.toArray().length==1)
        {
            drawRect = foreheadList.toArray()[0];
        }

        Point p1 = drawRect.tl();
        Point p2 = new Point(drawRect.width,drawRect.height);
        p1.x = (p1.x*2+subRect.tl().x)+20; p1.y = (p1.y*2+subRect.tl().y)+5;
        p2.x = (p2.x*2+p1.x)-40; p2.y = (p2.y*2+p1.y)-5;

        Imgproc.rectangle(foreheadFrame,p1,p2,new Scalar(0,255,0),2);

        center =new Point(((p1.x+p2.x)/2),((p1.y+p2.y)/2));
        Imgproc.circle(foreheadFrame,center,1,new Scalar(255,0,0),3);

        subMat = null;
    }
    void haFunction()
    {

//        sendData(center.x,center.y);
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

    static class DebugLog{
        public static boolean isStartDetect=false;
        public static String meanColorValue="";
        public static String currentColorValue="";
        public static String processTime="";
        public static String printLog(){
            return "isStartDetect: "+isStartDetect
                    +"\nmeanColorValue:"+meanColorValue
                    +"\ncurrentColorValue:"+currentColorValue
                    +"\nprocessTime:"+processTime;
        }
    }
}


