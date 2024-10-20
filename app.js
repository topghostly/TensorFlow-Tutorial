import * as tf from "@tensorflow/tfjs";

// while (true) {
//   tf.tidy(() => {
//     const values = [];
//     const shape = [5, 3];

//     for (let i = 0; i < 15; i++) {
//       values[i] = Math.random();
//     }

//     const tensorA = tf.tensor2d(values, shape);
//     const tensorB = tf.tensor2d(values, shape);

//     const tensorBTranspose = tf.transpose(tensorB);

//     const matMulTensor = tf.matMul(tensorA, tensorBTranspose);
//     console.log("Before dispose:", tf.memory().numTensors);
//   });
//   console.log("After dispose:", tf.memory().numTensors);
//   await new Promise((resolve) => setTimeout(resolve, 500));
// }

// // //
// USING THE LAYERS API
// // //

const model = tf.sequential();

// Hidden layer
model.add(
  tf.layers.dense({
    inputShape: [3],
    units: 4,
    activation: "relu",
  })
);

// Output layer
model.add(
  tf.layers.dense({
    units: 1,
    activation: "softmax",
  })
);

// Compile model
model.compile({
  optimizer: "adam",
  loss: "meanSquaredError",
});
