const slider = document.getElementById('molecule-range');
const output = document.getElementById('slider-value');
output.innerHTML = slider.value;

slider.oninput = function () {
  output.innerHTML = this.value;
  output.style.transform = 'scale(1.5)';
  setTimeout(function () {
    output.style.transform = 'scale(1)';
  }, 500);
}


document.querySelector('#form').addEventListener('submit', function (e) {
  console.log("hi")
  document.querySelector('#loadingScreen').classList.toggle("d-none")
});