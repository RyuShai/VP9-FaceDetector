package com.example.ryu.facedetectpreview;

import android.graphics.Color;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ryu on 11/23/17.
 */

public class SkinDetect {
    List<Mat> simpleColorKMeans(Mat img)
    {
        int K = 8;
        int n = img.rows() * img.cols();
        Mat data = img.reshape(1, n);
        data.convertTo(data, CvType.CV_32F);

        Mat labels = new Mat();
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT, 100, 1);
        Mat centers = new Mat();
//        kmeans(data, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);
        Core.kmeans(data,K,labels,criteria,1,Core.KMEANS_PP_CENTERS,centers);

        List<Mat> listMat = new ArrayList<Mat>();
        listMat.add(labels);
        listMat.add(centers);
        return listMat;
    }

}
