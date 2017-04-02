$(document).ready(function () {
  var img = new Array(784);
  for (var i=0; i<img.length; i++) {
    img[i] = 0;
  }
  var $cell = $('.cell').mousedown(function () {
        $(this).toggleClass('highlight');
        updateArray(this);
        var flag = $(this).hasClass('highlight')
        $cell.on('mouseenter.highlight', function () {
            $(this).toggleClass('highlight', flag);
            updateArray(this);
            $.ajax({
              type: 'POST',
              contentType: 'application/json',
              data: JSON.stringify(img),
              dataType: 'json',
              url: '/img',
              success: function (res) {
                handleUpdate(res.digit);
              }});
        });
    });
  $(document).mouseup(function () {
      $cell.off('mouseenter.highlight')
  });

  function updateArray(cell) {
    var row = parseInt($(cell).attr('data-row'));
    var col = parseInt($(cell).attr('data-col'));
    var idx = (28 * row) + col;
    img[idx] = 1;
  }

  function handleUpdate(digit) {
    var strings = ['Looks like a ', 'I think it\'s a ', 'The digit is probably a '];
    var newString = strings[Math.floor(Math.random() * strings.length)] + digit;
    $('#res').html(newString);
  }
});
