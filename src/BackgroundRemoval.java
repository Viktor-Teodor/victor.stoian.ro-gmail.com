import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public class BackgroundRemoval extends Application {

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

    /**
     * Get the average hue value of the image starting from its Hue channel
     * histogram
     *
     * @param hsvImg
     *            the current frame in HSV
     * @param hueValues
     *            the Hue component of the current frame
     * @return the average Hue value
     */
    private double getHistAverage(Mat hsvImg, Mat hueValues)
    {
        // init
        double average = 0.0;
        Mat hist_hue = new Mat();
        // 0-180: range of Hue values
        MatOfInt histSize = new MatOfInt(180);
        List<Mat> hue = new ArrayList<>();
        hue.add(hueValues);

        // compute the histogram
        Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));

        // get the average Hue value of the image
        // (sum(bin(h)*h))/(image-height*image-width)
        // -----------------
        // equivalent to get the hue of each pixel in the image, add them, and
        // divide for the image size (height and width)
        for (int h = 0; h < 180; h++)
        {
            // for each bin, get its value and multiply it for the corresponding
            // hue
            average += (hist_hue.get(h, 0)[0] * h);
        }

        // return the average hue of the image
        return average = average / hsvImg.size().height / hsvImg.size().width;
    }


    private Mat doBackgroundRemovalAbsDiff(Mat currFrame, Mat oldFrame) {
        Mat greyImage = new Mat();
        Mat foregroundImage = new Mat();

        if (oldFrame == null)
            oldFrame = currFrame;

        Core.absdiff(currFrame, oldFrame, foregroundImage);
        Imgproc.cvtColor(foregroundImage, greyImage, Imgproc.COLOR_BGR2GRAY);

        int thresh_type = Imgproc.THRESH_BINARY_INV;

        Imgproc.threshold(greyImage, greyImage, 10, 255, thresh_type);
        currFrame.copyTo(foregroundImage, greyImage);

        oldFrame = currFrame;
        return foregroundImage;

    }


    /**
     * Perform the operations needed for removing a uniform background
     *
     * @param frame
     *            the current frame
     * @return an image with only foreground objects
     */
    private Mat doBackgroundRemoval(Mat frame)
    {
        // init
        Mat hsvImg = new Mat();
        List<Mat> hsvPlanes = new ArrayList<>();
        Mat thresholdImg = new Mat();

        int thresh_type = Imgproc.THRESH_BINARY_INV;

        // threshold the image with the average hue value
        hsvImg.create(frame.size(), CvType.CV_8U);
        Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
        Core.split(hsvImg, hsvPlanes);

        // get the average hue value of the image
        double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));

        Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, thresh_type);

        Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));

        // dilate to fill gaps, erode to smooth edges
        Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);

        Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

        // create the new image
        Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        frame.copyTo(foreground, thresholdImg);

        return foreground;
    }

    public WritableImage loadAndConvert() throws Exception {

        List<Mat> planes = new ArrayList<>();
        // Loading the OpenCV core library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Imgcodecs imageCodecs = new Imgcodecs();

        String input = "testPen.jpg";
        String inputBackground = "testWithoutPen.jpg";

        // Reading the image
        Mat src = imageCodecs.imread(input);
        Mat background = imageCodecs.imread(inputBackground);

        Mat dest = doBackgroundRemovalAbsDiff(src,background);

        byte[] data1 = new byte[dest.rows() * dest.cols() * (int)(dest.elemSize())];
        dest.get(0, 0, data1);

        // Creating the buffered image
        BufferedImage bufImage = new BufferedImage(dest.cols(),dest.rows(),
                BufferedImage.TYPE_BYTE_GRAY);

        // Setting the data elements to the image
        bufImage.getRaster().setDataElements(0, 0, dest.cols(), dest.rows(), data1);

        // Creating a WritableImage
        WritableImage writableImage = SwingFXUtils.toFXImage(bufImage, null);
        System.out.println("Image Read");
        return writableImage;
    }

    public static void main(String args[]) throws Exception {
        launch(args);
    }
}
