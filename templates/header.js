// Function to create the header
function createHeader() {
  var header = document.createElement("header");

  // Home link
  var homeLink = document.createElement("a");
  homeLink.href = "index.html";
  homeLink.textContent = "Home";
  header.appendChild(homeLink);

  // Login link
  var loginLink = document.createElement("a");
  loginLink.href = "login.html";
  loginLink.textContent = "Login";
  header.appendChild(loginLink);

  // Logout link
  var logoutLink = document.createElement("a");
  logoutLink.href = "logout.html";
  logoutLink.textContent = "Logout";
  header.appendChild(logoutLink);

  // Go Back link
  var goBackLink = document.createElement("a");
  goBackLink.href = "javascript:history.back()";
  goBackLink.textContent = "Go Back";
  header.appendChild(goBackLink);

  return header;
}

// Function to add the header to the current page
function addHeader() {
  var body = document.getElementsByTagName("body")[0];
  var header = createHeader();
  body.insertBefore(header, body.firstChild);
}
