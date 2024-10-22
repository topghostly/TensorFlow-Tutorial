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
    units: 1,
    activation: "sigmoid",
  })
);

// Make optimiser
const optimiser = tf.train.sgd(0.1);

// Compile model
model.compile({
  optimizer: optimiser,
  loss: tf.losses.meanSquaredError,
});

// Make sone dummy data
const xs = tf.tensor2d([
  [0.5, 0.5],
  [0, 0],
  [1, 1],
]);

const ys = tf.tensor2d([[0.5], [1], [0]]);

const train = async () => {
  for (let i = 0; i < 10000; i++) {
    const responce = await model.fit(xs, ys, {
      epochs: 10,
      shuffle: true,
    });
    console.log(responce.history.loss[0]);
  }
};

const testData = tf.tensor2d([[0.6, 0.6]]);

train().then(() => {
  console.log("Training Complete");

  // Test the model with the input data
  const output = model.predict(testData);
  output.print();
});

// [[0.4661961],
// [0.821998 ],
// [0.2051105]]
