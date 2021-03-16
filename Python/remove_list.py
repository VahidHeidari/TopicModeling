#
# Unicode table:
# https://www.utf8-chartable.de/unicode-utf8-table.pl?start=1280&number=1024
# https://unicode-table.com/en/sets/punctuation-marks/
#

REMOVE_LIST = [
    # English characters
    u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I',
    u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
    u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z',

    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i',
    u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r',
    u's', u't', u'u', u'v', u'w', u'x', u'y', u'z',

    u'\u0299', u'\u0280', u'\u1d07',

    # Numbers
    u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'0',
    u'\u0660', u'\u0661', u'\u0662', u'\u0663', u'\u0664', u'\u0665', u'\u0666', u'\u0667', u'\u0668', u'\u0669',
    u'\u06F0', u'\u06F1', u'\u06F2', u'\u06F3', u'\u06F4', u'\u06F5', u'\u06F6', u'\u06F7', u'\u06F8', u'\u06F9',

    # Punctuations
    u'!', u'"' u'#', '$', u'%', u'&', u'\'', u'(', u')', u'*', u'+', u',', u'-', u'.', u'/',
    u'\u2013',
    u'\u061B', u'\u061E', u'\u061F', u'\u066A', u'\u066B', u'\u066C', u'\u066D',
    u'\u08FB', u'\u08FC', u'\u060C', u'\u2039', u'\u203a',
    u'\x22', u'\x27', u'\xab', u'\xbb', u'\x3a', u'\xB7', u'\xA0', u'\x7E',
    u'\x7D', u'\x7C', u'\x7B', u'\x60', u'\x5F', u'\x5E', u'\x5D', u'\x5C',
    u'\x5b', u'\x3f', u'\x3e', u'\x3b', u'\x3a', u'\x2f', u'\x2e', u'\x2c',
    u'\x2a', u'\x29', u'\x28', u'\x26', u'\x25', u'\x23', u'\x21',
]

