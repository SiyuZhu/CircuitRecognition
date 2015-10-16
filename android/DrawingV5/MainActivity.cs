using System;

using Android.App;
using Android.Content;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using Android.OS;
using Android.Graphics;
using System.IO;

namespace DrawingV5
{
	[Activity (Label = "DrawingV5", MainLauncher = true, Icon = "@drawable/icon")]
	public class MainActivity : Activity
	{

		protected override void OnCreate (Bundle bundle)
		{
			//Set Display/Interactions/Counter Vars
			base.OnCreate (bundle);
			SetContentView (Resource.Layout.Main);
			Button clear = FindViewById<Button> (Resource.Id.ClearButton);
			Button save = FindViewById<Button> (Resource.Id.SaveButton);
			int CountX = -7, CountY = -7;
			var IV = FindViewById<ImageView> (Resource.Id.DisplayView); 
			float x1=0, y1=0, x2=0, y2=0;
			Bitmap BM = Bitmap.CreateBitmap (760, 800, Bitmap.Config.Argb8888);

			//Set up Paint
			var P = new Paint();
			P.Color = Color.White;
			P.StrokeWidth = 15;
			Canvas C = new Canvas(BM);

			//Draw on Bitmap When Touched
			IV.Touch += (s, e) => {
				if (e.Event.Action == MotionEventActions.Down)
				{
					x1 = e.Event.GetX();
					y1 = e.Event.GetY();
					x2 = e.Event.GetX();
					y2 = e.Event.GetY();
					Console.WriteLine ("Touched at X:"+x1+" Y:"+y1);
					BM.SetPixel ((int)x1, (int)y1, Color.White);
					IV.SetImageBitmap (BM);
				}
				if (e.Event.Action == MotionEventActions.Move)
				{
					x1 = e.Event.GetX();
					y1 = e.Event.GetY();
					while(CountY<=7){
						while(CountX<=7){
							BM.SetPixel ((int)x1+CountX, (int)y1+CountY, Color.White);
							CountX++;
						}
						CountX=-7;
						CountY++;
					}
					Console.WriteLine ("Touched at X:"+x1+" Y:"+y1);
					C.DrawLine(x1, y1, x2, y2, P);
					IV.SetImageBitmap(BM);
					x2 = x1;
					y2 = y1;
				}
			};

			//Clear
			clear.Click += delegate {
				C.DrawARGB(255,0,0,0);
				IV.SetImageBitmap (BM);
			};


			//Save
			save.Click += delegate {
				ExportBitmapAsPNG(BM);
			};

		}

		void ExportBitmapAsPNG(Bitmap bitmap)
		{
			Random rnd = new Random();
			int RandAssign = rnd.Next(1, 1000000);
			var filePath = System.IO.Path.Combine("/storage/emulated/0/Pictures", "Drawing"+RandAssign+".png");
			var stream = new FileStream (filePath, FileMode.Create);
			bitmap.Compress(Bitmap.CompressFormat.Png, 100, stream);
			stream.Close();
			Console.WriteLine ("Saved");
		}
	}
}