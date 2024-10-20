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
    inputShape: [2],
    units: 4,
    activation: "sigmoid",
  })
);

// Output layer
model.add(
  tf.layers.dense({
    units: 3,
    activation: "sigmoid",
  })
);

// Compile model
const optimiser = tf.train.sgd(0.1);

model.compile({
  optimizer: optimiser,
  loss: tf.losses.cosineDistance,
});
