import tweepy
from threading import Timer
from truthy_twitter import *
import sys

def write(it, o):
    page = it.next()
    for id in page:
        o.write("%d\n" % id)

    time(it, o)

def time(it, o):
    t = Timer(11, write, args=[it, o])
    t.start()

if __name__ == "__main__":
    args = sys.argv

    if len(args) != 4:
        print "USAGE: python followers.py [id] [outfile] [oauth]"
        sys.exit()

    id = args[1]
    outfile = args[2]
    oauth = args[3]
    o = open(outfile, "wb")

    api = get_twitter_api(oauth)

    follower_cursors = tweepy.Cursor(api.followers_ids, id=id)

    it = follower_cursors.pages()
    write(it, o)





