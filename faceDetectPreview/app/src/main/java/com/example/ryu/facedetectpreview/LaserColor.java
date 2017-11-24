package com.example.ryu.facedetectpreview;

/**
 * Created by ryu on 11/23/17.
 */

public class LaserColor{
    private int red;
    private int green;
    private int blue;
    public LaserColor()
    {
        red=green=blue=0;
    }
    public LaserColor(int r, int g, int b)
    {
        setRed(r);
        setGreen(g);
        setBlue(b);
    }
    public String rgbString()
    {
        return String.valueOf(red)+" "+String.valueOf(green)+" "+String.valueOf(blue);
    }

    public int getRed() {
        return red;
    }

    public void setRed(int red) {
        this.red = red;
    }

    public int getGreen() {
        return green;
    }

    public void setGreen(int green) {
        this.green = green;
    }

    public int getBlue() {
        return blue;
    }

    public void setBlue(int blue) {
        this.blue = blue;
    }
}
