package com.example.ryu.facedetectpreview;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;

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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.opencv.core.Core.*;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener {
    private static String TAG = "Ryu_MainActivity";

    //
    JavaCameraView jCamera;
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
    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadesClassifier;
    private Mat grayScaleImage;
    private int absoluteFaceSize;
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
            cascadesClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
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

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        openCvCameraView = new JavaCameraView(this, -1);
        TextView txView = new TextView();
        openCvCameraView.addChildrenForAccessibility();
        setContentView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);
//        jCamera = (JavaCameraView) findViewById(R.id.javaCameraView);
//        jCamera.setVisibility(SurfaceView.VISIBLE);
//        jCamera.setCvCameraViewListener(this);
        // Example of a call to a native method
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(jCamera!=null)
            jCamera.disableView();
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
        if(OpenCVLoader.initDebug())
        {
            Log.d(TAG,"opencv loadded");
            mLoaderCallBack.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }else{
            Log.d(TAG,"opencv load failed");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, MainActivity.this , mLoaderCallBack);
        }
    }
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    @Override
    public void onCameraViewStarted(int width, int height) {
       grayScaleImage = new Mat(height,width,CvType.CV_8UC4);
        absoluteFaceSize = (int) (height*0.2);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(Mat inputFrame) {
        Imgproc.cvtColor(inputFrame,grayScaleImage,Imgproc.COLOR_RGBA2BGR);
        MatOfRect faces = new MatOfRect();
        //use classsifier to detect face
        if(cascadesClassifier != null)
        {
            cascadesClassifier.detectMultiScale(grayScaleImage,faces,1.1,2,2,
                    new Size(absoluteFaceSize,absoluteFaceSize), new Size());
        }
        Rect[] facesArray  = faces.toArray();
        for(int i=0; i<facesArray.length;i++){
            Imgproc.rectangle(inputFrame, facesArray[i].tl(),facesArray[i].br(),
                    new Scalar(0,255,0,255),3);
            Rect crop = new Rect();
            crop.x = facesArray[i].x;
            crop.y = facesArray[i].y;
            crop.height =facesArray[i].height;
            crop.width = facesArray[i].width;

            mRgba = new Mat(inputFrame,crop);
            ProcessWriteImage pwi = new ProcessWriteImage(this);
            pwi.execute();
        }
        return inputFrame;
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
        File sd = new File(Environment.getExternalStorageDirectory()+"/Ryu");
        boolean success = true;
        if(!sd.exists()){
            success = sd.mkdir();
        }
        if(success){
            File dest;
            do{
                fileName = sd.getAbsolutePath()+"/image"+iNum+".png";
                Log.d(TAG,"file path: "+ fileName);
                dest= new File(fileName);
                iNum++;
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
}


