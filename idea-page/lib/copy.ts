export async function copyToClipboard(value: string, fallbackMessage = "Copied") {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch (error) {
    console.error("Copy failed", error);
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const successful = document.execCommand("copy");
    document.body.removeChild(textarea);
    if (successful) {
      console.info(fallbackMessage);
      return true;
    }
    return false;
  }
}
