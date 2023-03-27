package com.example.nlpbertandroid;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;

import org.pytorch.Module;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.tensorflow.lite.DataType;

import org.tensorflow.lite.Interpreter;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private static Module module = null;
    private static Interpreter tfliteModule = null;
    private static final String MODELNAME = "mobilevit_xxs_2.tflite";

    private static final String PUBLIC_IMAGESET_FOLDER = "flickr/public/";
    private static final String PRIVATE_IMAGESET_FOLDER = "flickr/private/";

    private static final int RANDOM_LOWERBOUND = 1;
    private static final int RANDOM_UPPERBOUND = 200;
    private static final int RANDOM_IMAGESET_SIZE = 20;
    private static final double THRESHOLD = 0.02;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            initViews();
        } catch (Exception e) {
            Log.e("Error during startup", "Thrown from onCreate()", e);
            finish();
        }
    }

    //initialize views of the images and the buttons
    private void initViews() {
        try {
            if (tfliteModule == null)
                tfliteModule = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }
        findViewById(R.id.processButton).setOnClickListener(v -> {
            try {
                processImage();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    /*
    Utility function to load and read tflite file
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor=this.getAssets().openFd(MODELNAME);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }

    private void tfInputSize() {
        int inputIndex = 0;
        int outputIndex = 0;
        int[] inputShape = tfliteModule.getInputTensor(inputIndex).shape();
        DataType probabilityDataType =
                tfliteModule.getInputTensor(inputIndex).dataType();
        int[] outputShape = tfliteModule.getOutputTensor(outputIndex).shape();
        System.out.println(":Shape start:");
        System.out.println(inputShape[0]);
        System.out.println(inputShape[1]);
        System.out.println(inputShape[2]);
        System.out.println(inputShape[3]);
        System.out.println(probabilityDataType);
        System.out.println(outputShape[0]);
        System.out.println(outputShape[1]);
        System.out.println(":Shape end:");
    }

    private void processImage() throws IOException {

        //tfInputSize();

        // Variables needed to store train and test set for public and private classes
        List<Bitmap> publicBitmap = new ArrayList<>();
        List<Bitmap> privateBitmap = new ArrayList<>();
        Random random = new Random();
        int[] randomImageIndexes = random.ints(RANDOM_IMAGESET_SIZE, RANDOM_LOWERBOUND,
                RANDOM_UPPERBOUND).toArray();

        // Initialized AssetManager to access assets folder
        Context mContext = MainActivity.this;
        AssetManager assetManager = mContext.getAssets();
        String[] privateFiles = new String[0];
        String[] publicFiles = new String[0];
        try {
            privateFiles = assetManager.list(PRIVATE_IMAGESET_FOLDER);
            publicFiles = assetManager.list(PUBLIC_IMAGESET_FOLDER);
        } catch (IOException e) {
            e.printStackTrace();
        }
        List<String> privateFileList = new LinkedList<String>(Arrays.asList(privateFiles));
        List<String> publicFileList = new LinkedList<String>(Arrays.asList(publicFiles));

        // Initializing BitmapFactory object for handling preprocessed image bitmaps
        BitmapFactory.Options opt = new BitmapFactory.Options();
        Bitmap bm = null;
        InputStream is = null;

        for (int i=0; i<randomImageIndexes.length; i++){

            try {
                is = assetManager.open(PRIVATE_IMAGESET_FOLDER + privateFileList.
                        get(randomImageIndexes[i]));
                if (is!=null) {
                    bm = BitmapFactory.decodeStream(is, null, opt);
                    bm = Bitmap.createScaledBitmap(bm, 256,256,false);
                    if (bm != null) {
                        System.out.println("private bitmap:" + bm);
                        privateBitmap.add(bm);
                    }else
                        continue;
                } else {
                    continue;
                }


                is = assetManager.open(PUBLIC_IMAGESET_FOLDER +
                        publicFileList.get(randomImageIndexes[i]));
                if (is!=null) {
                    bm = BitmapFactory.decodeStream(is, null, opt);
                    bm = Bitmap.createScaledBitmap(bm, 256,256,false);
                    if (bm != null) {
                        System.out.println("public bitmap:" + bm);
                        publicBitmap.add(bm);
                    }
                    else
                        continue;
                } else {
                    continue;
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        float[][][][][] pubInput = new float[RANDOM_IMAGESET_SIZE][1][256][256][3];
        float[][][][][] priInput = new float[RANDOM_IMAGESET_SIZE][1][256][256][3];
        int i=-1;
        int batchNum = 0;

        // Analysis code for every frame
        // Preprocess the image
        for (Bitmap img: publicBitmap) {
            Bitmap bitmap= img;
            i++;

            for (int x = 0; x < 256; x++) {
                for (int y = 0; y < 256; y++) {
                    int pixel = bitmap.getPixel(x, y);
                    pubInput[i][batchNum][x][y][0] = (Color.red(pixel));
                    pubInput[i][batchNum][x][y][1] = (Color.green(pixel));
                    pubInput[i][batchNum][x][y][2] = (Color.blue(pixel));
                }
            }
        }

        i=-1;
        for (Bitmap img: privateBitmap) {

            Bitmap bitmap= img;
            i++;

            for (int x = 0; x < 256; x++) {
                for (int y = 0; y < 256; y++) {
                    int pixel = bitmap.getPixel(x, y);
                    priInput[i][batchNum][x][y][0] = (Color.red(pixel));
                    priInput[i][batchNum][x][y][1] = (Color.green(pixel));
                    priInput[i][batchNum][x][y][2] = (Color.blue(pixel));
                }
            }
        }

        // Initializing the output variable
        float[][] output=null;
        List<Integer> outputPublicPreds = new ArrayList<>();
        List<Integer> outputPrivatePreds = new ArrayList<>();

        // running the model
        for (float[][][][] floats: pubInput){
            output=new float[1][2];
            tfliteModule.run(floats,output);
//            System.out.println("0:"+output[0][0]);
//            System.out.println("1:"+output[0][1]);
            if (output[0][0]>THRESHOLD){
                outputPublicPreds.add(0);
            }else{
                outputPublicPreds.add(1);
            }
        }
        System.out.println("Public(1) preds: " + outputPublicPreds);

        for (float[][][][] floats: priInput){
            output=new float[1][2];
            tfliteModule.run(floats,output);
//            System.out.println("0:"+output[0][0]);
//            System.out.println("1:"+output[0][1]);
            if (output[0][0]>THRESHOLD){
                outputPrivatePreds.add(0);
            }else{
                outputPrivatePreds.add(1);
            }
        }
        System.out.println("Private(0) preds: " + outputPrivatePreds);

    }

}