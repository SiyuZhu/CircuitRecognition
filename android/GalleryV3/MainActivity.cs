using System;
using System.Runtime.InteropServices;
using Android.App;
using Android.Content;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using Android.OS;
using Android.Graphics;
using Android.Graphics.Drawables;
namespace GalleryV3
{
	[Activity (Label = "GalleryV3", MainLauncher = true, Icon = "@drawable/icon")]
	public class MainActivity : Activity
	{
		protected override void OnCreate (Bundle bundle)
		{
			base.OnCreate (bundle);
			SetContentView (Resource.Layout.Main);
			Button button = FindViewById<Button> (Resource.Id.myButton);
			button.Click += delegate {
				var imageIntent = new Intent ();
				imageIntent.SetType ("image/*");
				imageIntent.SetAction (Intent.ActionGetContent);
				StartActivityForResult (Intent.CreateChooser (imageIntent, "Select photo"), 0);
			};
		}

		//Actual Image Selector
		protected override void OnActivityResult (int requestCode, Result resultCode, Intent data)
		{
			base.OnActivityResult (requestCode, resultCode, data);
			if (resultCode == Result.Ok) {
				var IV = FindViewById<ImageView> (Resource.Id.myImageView);
				IV.SetImageURI (data.Data);
				IV.DrawingCacheEnabled = true;
				Bitmap BM = Bitmap.CreateBitmap (IV.GetDrawingCache(true));
				IV.DrawingCacheEnabled = false;
				TextView TV = FindViewById<TextView> (Resource.Id.DetailsText);
				TV.Text = (Pix(BM));
			}
		}

		//Calculate White Pixels pixels, Called above in OnActivityResult
		//Arbitrary function, will be replaced
		public string Pix (Bitmap BM) {
			//Calculate # of Pixels
			int x = 0;
			int y = 0;
			int WhiteCount = 0;
			while (y < BM.Height) {
				while (x < BM.Width) {
					if (BM.GetPixel (x, y) == -1) {
						WhiteCount++;
					}
					x++;
				}
				x = 0;
				y++;
			}
			return ("White Pixels:" + WhiteCount);
		}
	} 
}