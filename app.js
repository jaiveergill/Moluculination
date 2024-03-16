const express = require('express')
const cors = require('cors')
const app = express()
const path = require('path');
const {
  spawn
} = require('child_process');


app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, '/public')));



app.use(cors())


app.get("/", (req, res) => {
  res.render('index', {
    title: 'Home Page'
  });
})

app.get('/molecule', (req, res) => {
  const complexity = req.query.complexity;
  console.log('Molecule complexity:', complexity);



  var dataToSend;
  // spawn new child process to call the python script
  const python = spawn('python3', ['script.py', complexity]);
  // collect data from script
  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...');
    dataToSend = data.toString();
  });
  // in close event we are sure that stream from child process is closed
  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
    // send data to browser
    res.render('molecule.ejs', {
      smiles: dataToSend
    })
  });
});


app.listen(3000, () => {
  console.log('CORS-enabled web server listening on port 80')
})