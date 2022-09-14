package com.example.audiorecognition;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.intel.openvino.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.logging.Logger;

public class MainActivity extends AppCompatActivity {

    private static final Logger logger = Logger.getLogger(MainActivity.class.getName());

    // Audio recorder code is taken from Android tutorial: https://developer.android.com/guide/topics/media/mediarecorder
    private RecordButton recordButton = null;
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private boolean permissionToRecordAccepted = false;
    private String [] permissions = {Manifest.permission.RECORD_AUDIO};
    private AudioRecord recorder = null;

    // OpenVINO classes
    private Core core = null;
    private Thread inferenceThread = null;
    private CompiledModel compiledModel = null;
    private TextView resultView = null;

    private static final String[] labels = {
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "_silence_",
            "_unknown_"
    };

    private int maxScoreId;

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode){
            case REQUEST_RECORD_AUDIO_PERMISSION:
                permissionToRecordAccepted  = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                break;
        }
        if (!permissionToRecordAccepted ) finish();
    }

    private void onRecord(boolean start) {
        if (start) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    private void startRecording() {
        logger.info("Start recording");
        recorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                     16000,
                                   AudioFormat.CHANNEL_IN_MONO,
                                   AudioFormat.ENCODING_PCM_FLOAT,
                    16000 * 4);
        logger.info("state: " + recorder.getRecordingState() + " " + AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT));
        recorder.startRecording();
    }

    private void stopRecording() {
        logger.info("Stop recording");

        float[] data = new float[16000];
        recorder.read(data, 0, 16000, AudioRecord.READ_BLOCKING);

        recorder.stop();

        recorder.release();
        recorder = null;


        class InferenceRunnable implements Runnable {
            @Override
            public void run() {
                predict(data);
            }
        }
        inferenceThread = new Thread(new InferenceRunnable());
        inferenceThread.start();
    }

    class RecordButton extends Button {
        boolean mStartRecording = true;

        OnClickListener clicker = new OnClickListener() {
            public void onClick(View v) {
                onRecord(mStartRecording);
                if (mStartRecording) {
                    setText("Stop recording");
                } else {
                    setText("Start recording");
                }
                mStartRecording = !mStartRecording;
            }
        };

        public RecordButton(Context ctx) {
            super(ctx);
            setText("Start recording");
            setOnClickListener(clicker);
        }
    }

    private static String getResourcePath(InputStream in, String name, String ext) {
        String path = "";
        try {
            Path plugins = Files.createTempFile(name, ext);
            Files.copy(in, plugins, StandardCopyOption.REPLACE_EXISTING);
            path = plugins.toString();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return path;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);

        LinearLayout ll = new LinearLayout(this);
        ll.setOrientation(LinearLayout.VERTICAL);

        recordButton = new RecordButton(this);
        ll.addView(recordButton,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));

        resultView = new TextView(this);
        ll.addView(resultView,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));

        setContentView(ll);

        // Initialize OpenVINO
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Copy plugins.xml from resources
        InputStream in = Core.class.getClassLoader().getResourceAsStream("plugins.xml");
        String pluginsXml = getResourcePath(in, "plugins", "xml");

        // Create OpenVINO Core object using temporal plugins.xml
        core = new Core(pluginsXml);

        // Read model into memory
        String modelXml = "", modelBin = "";
        try {
            modelXml = getResourcePath(getAssets().open("ov_model.xml"), "ov_model", "xml");
            modelBin = getResourcePath(getAssets().open("ov_model.bin"), "ov_model", "bin");
        } catch (IOException ex) {
        }

        logger.info("Read OpenVINO model");
        Model model = core.read_model(modelXml, modelBin);

        logger.info("Compile OpenVINO model");
        compiledModel = core.compile_model(model, "CPU");
    }

    private void predict(float[] data) {
        logger.info("Create inference request");

        InferRequest ireq = compiledModel.create_infer_request();

        Tensor input = new Tensor(new int[]{1, 16000}, data);
        ireq.set_input_tensor(input);
        logger.info("Run inference");
        ireq.infer();
        logger.info("Finish inference");
        Tensor output = ireq.get_output_tensor();

        ireq.release();
        ireq = null;

        float[] outputData = output.data();
        maxScoreId = 0;
        for (int i = 1; i < outputData.length; ++i) {
            if (outputData[i] > outputData[maxScoreId])
                maxScoreId = i;
        }

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                resultView.setText("Command:" + labels[maxScoreId]);
            }
        });
    }
}
