import couchdb
import gzip
import csv
from models.dataset import DataSet

def write(tweet):
    print "saving tweet"

def read_followers(infile):
    followers = {}

    with open(infile, "rb") as in_fp:
        for row in in_fp:
            id = row.strip()
            followers[id] = 1

    return followers 

def filter_tweet(tweet):
    romney = read_followers("romney_followers.csv")
    barack = read_followers("barack_followers.csv")
    partisanship = 0
    found = 0

    id = int(tweet["user"]["id"])
    if id in romney:
        partisanship += 1
        found = 1

    if id in barack:
        partisanship -= 1
        found = 1

    if found:
        res = {}
        res["text"] = tweet["text"]
        res["partisanship"] = partisanship
        res["screen_name"] = tweet["user"]["screen_name"]
        res["id"] = tweet["id"]
        res["user_id"] = tweet["user"]["id"]
        res["created_at"] = tweet["created_at"]
        fp.write(str(res))
        return True
    else:
        return False


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-s', '--start', dest='start',
                      help='start date',
                      default="2012-08-01")
    parser.add_option('-e', '--end', dest='end',
                      help='end date, exclusive.',
                      default="2012-11-02")
    (options, args) = parser.parse_args()


    couch = couchdb.Server("http://rissarae.net:5984")
    dbname = "truthy"

    fp = gzip.open("tweets.json.gz", "wb")

    db = couch[dbname]

    ds = DataSet(dbname) 
    ds.filter = filter_tweet
    ds.write = write

    ds.collect(options.start, options.end)

    fp.close()



