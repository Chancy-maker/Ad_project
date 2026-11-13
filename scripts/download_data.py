"""Télécharge les datasets publics lorsque c'est possible, sinon affiche les étapes manuelles."""
import _bootstrap  # noqa: F401

from ad_optimizer.data.download import download_all

if __name__ == "__main__":
    for name, st in download_all().items():
        print(f"\n[{name}] {st['status']}")
        if st.get("message"):
            print("  ", st["message"])
        if st.get("error"):
            print("   (erreur technique :", st["error"], ")")
