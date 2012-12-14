import couchdb
import gzip
import csv
from models.dataset import DataSet

def write(tweet):
    print "saving tweet"
    fp.write(str(tweet))

def read_followers():
    followers = {}

    with open("romney_followers", "rb") as in_fp:
        reader = csv.reader(in_fp, delimiter="\t")
        for row in reader:
            sn = row[0].strip()
            partisanship = row[1].strip()

            partisans[sn] = partisanship

    return partisans

def filter_tweet(tweet):
    partisans = read_partisans("partisans.csv")
    sn = tweet["user"]["screen_name"]
    if sn in partisans:
        res = {}
        res["text"] = tweet["text"]
        res["partisanship"] = partisans[sn]
        res["screen_name"] = tweet["user"]["screen_name"]
        res["id"] = tweet["id"]
        res["user_id"] = tweet["user"]["id"]
        res["created_at"] = tweet["created_at"]
        db.save(res)
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


    couch = couchdb.Server("http://krmckelv@l3v3l:127.0.0.1:5984")
    dbname = "truthy_measure"

    fp = gzip.open("tweets.json.gz", "wb")

    db = couch[dbname]

    ds = DataSet(dbname) 
    ds.filter = filter_tweet
    ds.write = write

    ds.collect(options.start, options.end)

    fp.close()



