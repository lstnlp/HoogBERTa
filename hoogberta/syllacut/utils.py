import re

def replace_emoji(text, token='<EMOJI>'):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64f"  # emoticons
        u"\U0001F300-\U0001F5ff"  # symbols & pictographs
        u"\U0001F680-\U0001F6ff"  # transport & map symbols
        u"\U0001F1e0-\U0001F1ff"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(token, text)

def normalize_quotes(text):
    # define the pattern for non-ASCII quotes
    quote_pattern = re.compile(u'[\u2018\u2019\u201c\u201d\u00ab\u00bb\u201e\u201f\u2039\u203a]')
    # replace non-ASCII quotes with ASCII single quotes
    text = quote_pattern.sub("'", text)
    # define the pattern for non-ASCII double quotes
    double_quote_pattern = re.compile(u'[\u201c\u201d\u00ab\u00bb\u201e\u201f\u2039\u203a]')
    # replace non-ASCII double quotes with ASCII double quotes
    text = double_quote_pattern.sub('"', text)
    return text

def add_space_around_symbols(text):
    # define the pattern for special symbols
    symbol_pattern = re.compile(r'([\[\]!@#\$%\^&\*\(\);:\'\"\?<>\{\}\|\/\\\+=-])')
    # add space before and after the symbols
    return symbol_pattern.sub(r' \1 ', text)

def add_space_around_comma_and_dot(text):
    # define the pattern for special symbols
    symbol_pattern = re.compile(r'([,\.])')
    # add space before and after the symbols
    return symbol_pattern.sub(r' \1 ', text)

def add_space_around_thai(text):
    # define the pattern for Thai characters
    thai_pattern = re.compile(u'([\u0E00-\u0E7F]+)')
    # add space before and after the Thai characters
    return thai_pattern.sub(r' \1 ', text).strip()

def convert_thai_to_arabic(thai_num):
    thai_nums = {'๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4', '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'}
    arabic_num = ''
    for char in thai_num:
        if char in thai_nums:
            arabic_num += thai_nums[char]
        else:
            arabic_num += char
    return arabic_num

def find_and_replace_links(text):
    pattern = r'(http[s]?://|ftp://)([a-zA-Z0-9\.\-]+)([/a-zA-Z0-9\.\-]*)'
    links = []
    def replace_links(match):
        links.append(match.group(0))
        return "<LINK>"
    replaced_text = re.sub(pattern, replace_links, text)
    return replaced_text, links

def find_number(text):
    p2 = re.compile(r'\b([0-9]+|[0-9]{1,3}(,[0-9]{3})*)(.[0-9]+)?\b')
    text = p2.sub("<NUM>", text)
    text = text.replace("<NUM>.<NUM>","<NUM>")
    return text

def replace_non_thai(text):
    # define the pattern for non-Thai characters
    non_thai_pattern = re.compile(r'([^\u0E00-\u0E7F\[\]\ \.,!@#\$%\^&\*\(\);:\'\"\?<>\{\}\|\/\\\+=-^0-9]+)')
    # replace non-Thai sequences with <ENG>
    return non_thai_pattern.sub(r' <wENG> ', text)

def find_non_thai(text):
    non_thai_pattern = re.compile(r'([^\u0E00-\u0E7F\[\]\ \.,!@#\$%\^&\*\(\);:\'\"\?<>\{\}\|\/\\\+=-^0-9]+)')
    # replace non-Thai sequences with <ENG>
    return non_thai_pattern.finditer(text)

def pipeline1(text):
    text = text.lower()
    text = replace_emoji(text, token='EMOJI')
    text = " ".join(add_space_around_symbols(text).split())
    text = " ".join(add_space_around_thai(text).split())
    text = convert_thai_to_arabic(text)
    text = normalize_quotes(text)
    text = replace_non_thai(text)
    text = find_number(text)
    print(text)
    text = " ".join(add_space_around_comma_and_dot(text).split())
    return text    

def test_all():
    out = replace_emoji("HELLO 😀😀 How are you 😀", token='<EMOJI>')
    print(out)

    out = add_space_around_symbols("Hi[hello] I'm a stu_dent.")
    print(out)

    out = add_space_around_thai("ฉันต้อง(การกระดาษ)เอ4")
    print(out.strip())

    out = normalize_quotes("ฉันอยากได้ ‘ไอแพด‘ ใหม่")
    print(out)

    out = convert_thai_to_arabic("ปี ๒๕๖๖")
    print(out)

    out = find_number(out)
    print(out)

    out = replace_non_thai("ฉันอยากได้ ' กกกipad : 我们 @ ใหม่")
    print(out)

    out = pipeline1("ฉันอยากได้ 😀 ' เครื่องIpad Ipad : 我们@ใหม่ A:12,000.1 เครื่อง")
    print(out)

if __name__ == "__main__":
    match = find_non_thai("ก ipad 1234 กขคง nonthai ok")
    for m in match:
        print(m)

