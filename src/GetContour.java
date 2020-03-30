import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.opencv.videoio.VideoCapture;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.minMaxLoc;

public class GetContour extends Application{

    //ATTENTION! all distances are in millimeters

    //Just look here https://www.scantips.com/lights/subjectdistance.html
    private static double distanceToTheObject=235;
    private static double sensorWidth = 3.58, sensorHeight = 2.02, focalLength = 4.2, pixelsSensorHeight = 720;
    private static double pixelSensorWidth = 720;
    private static Point pixelUnderCutter = new Point(620,235);

    @Override
    public void start(Stage stage) throws Exception {
        WritableImage writableImage = loadAndConvert();

        // Setting the image view
        ImageView imageView = new ImageView(writableImage);

        // Setting the position of the image
        imageView.setX(10);
        imageView.setY(10);

        // setting the fit height and width of the image view
        imageView.setFitHeight(400);
        imageView.setFitWidth(600);

        // Setting the preserve ratio of the image view
        imageView.setPreserveRatio(true);

        // Creating a Group object
        Group root = new Group(imageView);

        // Creating a scene object
        Scene scene = new Scene(root, 600, 400);

        // Setting title to the Stage
        stage.setTitle("Reading image as grayscale");

        // Adding scene to the stage
        stage.setScene(scene);

        // Displaying the contents of the stage
        stage.show();
    }

    public WritableImage loadAndConvert() throws Exception {

        // Loading the OpenCV core library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Imgcodecs imageCodecs = new Imgcodecs();

        String modelWeights = "yolov3.weights"; //Download and load only wights for YOLO , this is obtained from official YOLO site//
        String modelConfiguration = "yolov3.cfg";//Download and load cfg file for YOLO , can be obtained from official site//
        Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights); //OpenCV DNN supports models trained from various frameworks like Caffe and TensorFlow. It also supports various networks architectures based on YOLO//

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = getOutputNames(net);

        String input = "dog.jpg";

        // Reading the image
        Mat colouredInput = imageCodecs.imread(input);
        //Mat colouredInput = captureImage();
        //Imgproc.cvtColor(colouredInput,colouredInput,Imgproc.COLOR_RGB2BGR);
        Mat src = new Mat(colouredInput.size(), CvType.CV_8UC1);
        Imgproc.cvtColor(colouredInput, src,Imgproc.COLOR_RGB2GRAY);


        //gaussian and mean blur, adaptive thresholding
        Mat imageA = prepareForProcessing(src);

        //apply Canny edge detection to find the edges
        Mat edges = applyCanny(imageA,20);

        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();

        //find the contours of the canny edges and apply it to the original image
        Mat contour = findContoursAfterCanny(edges, contours, hierarchy);

        POJOForEncapsulatingSquare square = findSize(contour,contours,colouredInput);

        Mat newColouredInput = new Mat(600,600,CvType.CV_8UC3);
        Imgproc.resize(colouredInput,newColouredInput, new Size(600,600),0,0,Imgproc.INTER_LINEAR);

        Mat blob = Dnn.blobFromImage(newColouredInput, 1.0 / 255, new Size(416,416), new Scalar(0), true, false,CvType.CV_32F); // We feed one frame of video into the network at a time, we have to convert the image to a blob. A blob is a pre-processed image that serves as the input.//
        net.setInput(blob);
        net.forward(result, outBlobNames); //Feed forward the model to get output //

        float threshold = 0.5f;       //for confidence
        float nmsThreshold = 0.3f;    //threshold for nms


        Mat dest = new Mat();
        GetResult(result, colouredInput, threshold, nmsThreshold,true);

        byte[] data1 = new byte[dest.rows() * dest.cols() * (int)(dest.elemSize())];
        dest.get(0, 0, data1);

        // Creating the buffered image
        BufferedImage bufImage = new BufferedImage(dest.cols(),dest.rows(), BufferedImage.TYPE_3BYTE_BGR);

        // Setting the data elements to the image
        bufImage.getRaster().setDataElements(0, 0, dest.cols(), dest.rows(), data1);

        // Creating a WritableImage
        WritableImage writableImage = SwingFXUtils.toFXImage(bufImage, null);
        System.out.println("Image Read");

        return writableImage;

    }

    private void GetResult(List<Mat> output, Mat image, float threshold, float nmsThreshold, boolean nms) {

        ArrayList<Integer> classIds = new ArrayList<>();
        ArrayList<Double> confidences = new ArrayList<>();
        ArrayList<Double> probabilities = new ArrayList<>();
        List<RotatedRect> boxes = new ArrayList<>();

            int w = image.cols();
            int h = image.rows();

            /*
             YOLO3 COCO trainval output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~ 84 : class probability
            */

            int prefix = 5;   //skip 0~4

            for (int k = 0; k < output.size(); k++) {
                Mat prob = output.get(k);
                for (int i = 0; i < prob.rows(); i++) {
                    double confidence = prob.get(i, 4)[0];

                    if (confidence > threshold) {
                        //get classes probability
                        Core.MinMaxLocResult minAndMax = minMaxLoc(prob);
                        int classes = (int) minAndMax.maxLoc.x;
                        double probability = prob.get(i, classes + prefix)[0];

                        if (probability > threshold) //more accuracy, you can cancel it
                        {
                            //get center and width/height
                            double centerX = prob.get(i, 0)[0] * w;
                            double centerY = prob.get(i, 1)[0] * h;
                            double width = prob.get(i, 2)[0] * w;
                            double height = prob.get(i, 3)[0] * h;

                            if (!nms) {
                                // draw result (if don't use NMSBoxes)
                                //Draw(image, classes, confidence, probability, centerX, centerY, width, height);
                                continue;
                            }

                            //put data to list for NMSBoxes
                            classIds.add(classes);
                            confidences.add(confidence);
                            probabilities.add(probability);
                            Point centre = new Point(centerX, centerY);
                            Size size = new Size(width, height);
                            boxes.add(new RotatedRect(centre, size, 0));
                        }
                    }
                }
            }

            //using non-maximum suppression to reduce overlapping low confidence box
            int[] indices = new int[]{8, 8, 8, 8, 8, 8, 8, 8};
            double[] con = new double[confidences.size()];

            for (int i = 0; i < confidences.size(); i++) {
                con[i] = confidences.get(i);
            }

    }

    private static Mat detectObject(Mat src, List<Mat> result){

        float confThreshold = 0.3f; //Insert thresholding beyond which the model will detect objects//
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();

        for (int i = 0; i < result.size(); ++i)
        {
            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities
            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j)
            {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = minMaxLoc(scores);
                float confidence = (float)mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold)
                {
                    int centerX = (int)(row.get(0,0)[0] * src.cols()); //scaling for drawing the bounding boxes//
                    int centerY = (int)(row.get(0,1)[0] * src.rows());
                    int width   = (int)(row.get(0,2)[0] * src.cols());
                    int height  = (int)(row.get(0,3)[0] * src.rows());
                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    clsIds.add((int)classIdPoint.x);
                    confs.add((float)confidence);
                    rects.add(new Rect(left, top, width, height));
                }
            }
        }

        float nmsThresh = 0.5f;
        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
        Rect[] boxesArray = rects.toArray(new Rect[0]);
        MatOfRect boxes = new MatOfRect(boxesArray);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); //We draw the bounding boxes for objects here//

        int [] ind = indices.toArray();
        int j=0;
        for (int i = 0; i < ind.length; ++i)
        {
            int idx = ind[i];
            Rect box = boxesArray[idx];
            Imgproc.rectangle(src, box.tl(), box.br(), new Scalar(0,0,255), 2);
            //i=j;

            System.out.println(idx);
        }

        return src;
    }

    private static List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));//unfold and create R-CNN layers from the loaded YOLO model//
        return names;
    }

    public static Point midPoint(Point a, Point b){
        return new Point((a.x+b.x)/2,(a.y+b.y)/2);
    }

    public static List<Point> orderCorners(Mat corners){
        // order the corners ascending by the x coordinate (negative values are to the top of the image)
        List<Point> orderedCorners =  new ArrayList<>();
        Point aux;

       orderedCorners.add(new Point(corners.get(0,0)[0],corners.get(0,1)[0]));
       orderedCorners.add(new Point(corners.get(1,0)[0],corners.get(1,1)[0]));
       orderedCorners.add(new Point(corners.get(2,0)[0],corners.get(2,1)[0]));
       orderedCorners.add(new Point(corners.get(3,0)[0],corners.get(3,1)[0]));

        for(int i = 0 ; i<3; ++i){
            for(int j = i+1; j<=3; ++j){
                if(orderedCorners.get(i).y > orderedCorners.get(j).y){
                    aux = orderedCorners.get(i);
                    orderedCorners.set(i,orderedCorners.get(j));
                    orderedCorners.set(j,aux);
                }
            }
        }


        Point topLeft ;
        Point topRight;
        Point bottomRight;
        Point bottomLeft;

        //the first two points will be the top corners, we just need to decide which is right and which is left
        if(orderedCorners.get(0).x< orderedCorners.get(1).x){
                topLeft = orderedCorners.get(0);
                topRight = orderedCorners.get(1);
        }
        else{
            topLeft = orderedCorners.get(1);
            topRight = orderedCorners.get(0);
        }

        if(orderedCorners.get(2).x< orderedCorners.get(3).x){
            bottomLeft = orderedCorners.get(2);
            bottomRight = orderedCorners.get(3);
        }
        else{
            bottomLeft = orderedCorners.get(3);
            bottomRight = orderedCorners.get(2);
        }

        orderedCorners.clear();
        orderedCorners.add(topLeft);
        orderedCorners.add(bottomLeft);
        orderedCorners.add(bottomRight);
        orderedCorners.add(topRight);

        return orderedCorners;
    }

    public static double euclideanDistance(Point a, Point b){
        return Math.sqrt(Math.pow(a.x-b.x,2) + Math.pow(a.y-b.y,2));
    }

    public static POJOForEncapsulatingSquare findSize(Mat src, List<MatOfPoint> contours, Mat originalColored){

        POJOForEncapsulatingSquare rectangle = new POJOForEncapsulatingSquare();
        Mat original = new Mat(); Mat corners = new Mat();

        src.copyTo(original);
        MatOfPoint2f coord = new MatOfPoint2f( contours.get(0).toArray());
        RotatedRect box = Imgproc.minAreaRect(coord);
        Imgproc.boxPoints(box,corners);

        List<Point> orderedCorners;
        orderedCorners = orderCorners(corners);

        Point topLeft = orderedCorners.get(0);
        Point bottomLeft = orderedCorners.get(1);
        Point bottomRight = orderedCorners.get(2);
        Point topRight = orderedCorners.get(3);

        Imgproc.line(originalColored,topLeft,bottomLeft, new Scalar(255,0,0),5);
        Imgproc.line(originalColored,bottomLeft,bottomRight, new Scalar(255,0,0),5);
        Imgproc.line(originalColored,bottomRight,topRight, new Scalar(255,0,0),5);
        Imgproc.line(originalColored,topRight,topLeft, new Scalar(255,0,0),5);

        Point topMid = midPoint(topLeft,topRight);
        Point bottomMid  = midPoint(bottomLeft,bottomRight);
        Point leftMid = midPoint(topLeft,bottomLeft);
        Point rightMid = midPoint(bottomRight,topRight);

        Imgproc.line(src, topMid,bottomMid, new Scalar(255,0,0));
        Imgproc.line(src, leftMid,rightMid, new Scalar(255,0,0));

        double objectHeightOnSensor = (sensorHeight * euclideanDistance(topMid,bottomMid))/420;
        double realObjectHeight = (distanceToTheObject * objectHeightOnSensor)/focalLength;

        double objectWidthOnSensor = (sensorWidth * euclideanDistance(leftMid,rightMid))/720;
        double realObjectWidth = (distanceToTheObject * objectWidthOnSensor)/focalLength;

        rectangle.bottomRight = bottomRight;
        rectangle.bottomLeft = bottomLeft;
        rectangle.topRight = topRight;
        rectangle.topLeft = topLeft;
        rectangle.height = realObjectHeight;
        rectangle.width = realObjectWidth;

        objectWidthOnSensor = (sensorWidth * getHorizontalDistance((topRight.x > bottomRight.x)? topRight : bottomRight
                                                                ,pixelUnderCutter))/720;
        realObjectWidth = (distanceToTheObject * objectWidthOnSensor)/focalLength;

        rectangle.distanceToCutter = realObjectWidth;

        return rectangle;
    }

    public static double getHorizontalDistance(Point topOfObject, Point pointUnderPixel){
        return Math.sqrt(Math.pow(topOfObject.x-pointUnderPixel.x,2));
    }

    public static Mat prepareForProcessing(Mat src){
        Mat imageBlurr = new Mat(src.size(), CvType.CV_8UC4);
        //Mat imageA = new Mat(src.size(), CvType.CV_32F);

        Imgproc.GaussianBlur(src, imageBlurr, new Size(7,7), 0);
        //Imgproc.adaptiveThreshold(imageBlurr, imageA, 255,Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY,7, 5);

        return imageBlurr;
    }

    public static Mat applyCanny(Mat src, float threshLow){

        Mat detectedEdges = new Mat();
        Mat kernel = Mat.ones(5, 5, CvType.CV_8UC1);

        Imgproc.Canny(src, detectedEdges, threshLow, threshLow * 3);

        Mat output = new Mat();

        //copy only the pixels that are on the edge
        src.copyTo(output, detectedEdges);

        //apply "close" operation to reduce noise
        Mat closing = new Mat();
        Imgproc.morphologyEx(output, closing,Imgproc.MORPH_DILATE,kernel);
        Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_DILATE,kernel);
        //Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_DILATE,kernel);
        //Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_DILATE,kernel);


       // Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_ERODE,kernel);
        //Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_ERODE,kernel);
        Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_ERODE,kernel);
        Imgproc.morphologyEx(closing, closing,Imgproc.MORPH_ERODE,kernel);


        return closing;
    }

    public static Mat findContoursAfterCanny(Mat src,List<MatOfPoint> contours, Mat hierarchy){

        Imgproc.findContours(src,contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea= -1;
        MatOfPoint maxContour= new MatOfPoint();

        for (int contour =0; contour< contours.size();++contour) {
            double area = Imgproc.contourArea(contours.get(contour));
            if (Imgproc.contourArea(contours.get(contour)) > maxArea) {
                maxArea=area;
                maxContour=contours.get(contour);
            }
        }

        contours.clear();
        contours.add(maxContour);

        Imgproc.drawContours(src,contours,-1,new Scalar(255,0,0));

        return src;
    }

    public static Mat captureImage(){

        Mat frame = new Mat();
        VideoCapture camera = new VideoCapture(1);
        if(!camera.isOpened()){
            System.out.println("Error");
        }
        else {

            camera.read(frame);
        }

        camera.release();
        return frame;
    }

    public static void main(String args[]) throws Exception {
        launch(args);
    }


}
