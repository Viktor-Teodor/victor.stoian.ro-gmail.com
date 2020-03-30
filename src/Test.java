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

public class Test extends Application {

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

        List<Mat> planes = new ArrayList<>();
        // Loading the OpenCV core library
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );

        // Instantiating the imagecodecs class
        Imgcodecs imageCodecs = new Imgcodecs();

        String input = "testPen.jpg";

        // Reading the image
        Mat src = imageCodecs.imread(input, Imgcodecs.IMREAD_GRAYSCALE);

        //Make the frequency space
        Mat detectedEdges = new Mat();

        // reduce noise with a 3x3 kernel
        Imgproc.blur(src, detectedEdges, new Size(5, 5));

        // canny detector, with ratio of lower:upper threshold of 3:1
        Imgproc.Canny(detectedEdges, detectedEdges, 0.3, 0.3 * 3);

        // using Canny's output as a mask, display the result
        Mat dest = new Mat();
        src.copyTo(dest, detectedEdges);
        //Mat kernel =Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2, 2));

        //Imgproc.erode(dest, dest,kernel);

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
