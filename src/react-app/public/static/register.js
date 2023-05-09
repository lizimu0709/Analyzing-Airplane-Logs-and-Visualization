function bindEmailCaptchaClick() {
  $("#captcha-btn").click(function (event) {
    // $this: represents the current button's jQuery object
    var $this = $(this);
    // Prevent the default event
    event.preventDefault();

    var email = $("input[name='email']").val();
    $.ajax({
      // http://127.0.0.1:500
      // /auth/captcha/email?email=xx@qq.com
      url: "/captcha/email?email=" + email,
      method: "GET",
      success: function (result) {
        var code = result['code'];
        if (code == 200) {
          var countdown = 20;
          // Before starting the countdown, cancel the button's click event
          $this.off("click");
          var timer = setInterval(function () {
            $this.text(countdown);
            countdown -= 1;
            // When the countdown ends
            if (countdown <= 0) {
              // Clear the timer
              clearInterval(timer);
              // Change the button's text back
              $this.text("Get Verification Code");
              // Rebind the click event
              bindEmailCaptchaClick();
            }
          }, 1000);
          // alert("Email verification code sent successfully!");
        } else {
          alert(result['message']);
        }
      },
      fail: function (error) {
        console.log(error);
      }
    });
  });
}

// Executed after the entire web page has been fully loaded
$(function () {
  bindEmailCaptchaClick();
});