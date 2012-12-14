import tweepy, time
import datetime

accounts = {
    "truthy" : { 'consumer_key' : 'NJ2HjrTVdbMV3kuC3kX3pg',
                 'consumer_secret' : 'FITXXKiHMlzqJFnmmGZWsINvps4cVnaBauw3cpUZV4U',
                 'token_key' : '187521608-k2yLmEAwmEKa9JlMypSzIbdHNpVuzCBNATFCAghD',
                 'token_secret' : 'sAXSXZoHWmnmHI3xVwN3FG5lRHLJj8S2pNxLgA'
                 },
    "bruno" : { 'consumer_key' : 'yjoQ1xC8ONdZXFbf8ikwFA',
                'consumer_secret' : 'mm0AX53Neh3k4EHuQGVYEkkYwbZyJ1Scu765sKDfjo',
                'token_key' : '15008596-KRHzycHMgThNkfuvodGtStc9bTzAuw103goAx9lFc',
                'token_secret' : 'HyIsGe5zj4tETNA2f33TlCVwhcRwb5NJjLztAt5I'
                },                

    "bruno2" : { 'consumer_key' : 'NQivw46sCpWDaSIuNBOQ',
                 'consumer_secret' : 'AprqzFznbzGzPaSHsWjixKrseL9e9y5GfMID73avJ8M',
                 'token_key' : '131553174-TVYh1mr1ftcfA0RQ88q6mqPVBUWgXkTTb2KAeyZM',
                 'token_secret' : 'Xx5NrqCMgB1uloZlNmZ2UqQfoLHkNFwjwyQjQMfDKY'
                },
    "truthy_system" : {
        'consumer_key' : '7h5UMDNblOHxzhVu3cIMg',
        'consumer_secret' : 'hYy42HiVKCD6BhzBkAOOIKq43H74t70qvEhwwC5WE',
        'token_key' : '206352159-Hefp19Be0j8wqY3Jr3RIpcFY2PkW1CdIHghFsTCY',
        'token_secret' : '8ETwkvAhYfQk2CtozM5haxM67gQYS0OdJWRdBRjM'
        }
    }

# setting up new oauth credentials:
#  Open up a new python interpreter to use interactively for the following.
#
#  1) get consumer key and consumer secret from http://dev.twitter.com/apps/new
#
#  2) import tweepy
#     auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#
#  3) auth.get_authorization_url()
#
#     That prints a URL. Visit the URL in your browser, grant access, and get the
#     verifier code.
#
#  4) auth.get_access_token(verifier_code)
#
#  5) now get the goodies:
#
#      auth.access_token.key
#      auth.access_token.secret
#

######################################################################
def get_twitter_api(which_account="truthy"):
    if which_account not in accounts:
        print "Unknown account:", which_account
        return None

    print "Using twitter configuration '%s'" % which_account
    
    dat = accounts[which_account]
    
    auth = tweepy.OAuthHandler(dat['consumer_key'], dat['consumer_secret'])
    auth.set_access_token(dat['token_key'], dat['token_secret'])

    api = tweepy.API(auth_handler=auth, secure=True, retry_count=3)
    return api

