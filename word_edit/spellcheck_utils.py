import re

reg_ex = re.compile(r"^[a-z][a-z]*[a-z]$")
no_reg_ex = re.compile(r".*[0-9].*")
mc_reg_ex = re.compile(r".*[A-Z].*[A-Z].*")


def containsNumber(text):
    """包含数字."""
    return no_reg_ex.match(text)


def containsMultiCapital(text):
    """包含多个大写字母."""
    return mc_reg_ex.match(text)


def can_spellcheck(w: str):
    """检查是否需要进行拼写检查."""
    # return not ((not reg_ex.match(w)) or containsMultiCapital(w) or containsNumber
    if reg_ex.match(w):
        return True
    else:
        return False
